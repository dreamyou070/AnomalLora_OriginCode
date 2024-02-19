import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
import torch
import os
from sub.mvtec_sy import MVTecDRAEMTrainDataset
from model.diffusion_model import load_target_model, transform_models_if_DDP
from model.lora import create_network
from attention_store import AttentionStore
from model.tokenizer import load_tokenizer
from utils import get_epoch_ckpt_name, save_model, prepare_dtype
from utils.accelerator_utils import prepare_accelerator
from sub.attention_control import register_attention_control
from utils.optimizer_utils import get_optimizer, get_scheduler_fix
from utils.model_utils import prepare_scheduler_for_custom_training, get_noise_noisy_latents_partial_time
from model.unet import unet_passing_argument
from sub.attention_control import passing_argument

vae_scale_factor = 0.18215

def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

def call_unet(args, accelerator, unet, noisy_latents, timesteps,
              text_conds, batch, weight_dtype, trg_indexs_list, mask_imgs):
    noise_pred = unet(noisy_latents,
                      timesteps,
                      text_conds,
                      trg_layer_list=args.trg_layer_list,
                      noise_type=args.noise_type).sample
    return noise_pred


def main(args):
    print(f'\n step 1. setting')
    output_dir = args.output_dir
    print(f' *** output_dir : {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(args.logging_dir, exist_ok=True)
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    if args.seed is None: args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)

    print(f'\n step 2. dataset')
    tokenizer = load_tokenizer(args)
    tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]
    obj_dir = os.path.join(args.data_path, args.obj_name)
    train_dir = os.path.join(obj_dir, "train")
    root_dir = os.path.join(train_dir, "good/rgb")
    if args.anomal_src_more:
        args.anomaly_source_path = os.path.join(args.data_path, "anomal_source_more")
    else :
        args.anomaly_source_path = os.path.join(args.data_path, "anomal_source")
    dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption=args.trigger_word,
                                     use_perlin=True,
                                     num_repeat=args.num_repeat,
                                     anomal_only_on_object=args.anomal_only_on_object,
                                     anomal_training  = True,
                                     latent_res = args.latent_res,
                                     perlin_max_scale = args.perlin_max_scale,
                                     kernel_size = args.kernel_size,)

    print(f'\n step 3. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 4. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    vae_dtype = weight_dtype
    print(f' (4.1) base model')
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype, accelerator)
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    print(' (4.2) lora model')
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value
    network = create_network(1.0, args.network_dim, args.network_alpha,
                             vae, text_encoder, unet, neuron_dropout=args.network_dropout, **net_kwargs, )
    train_unet, train_text_encoder = args.train_unet, args.train_text_encoder
    network.apply_to(text_encoder, unet, train_text_encoder, train_unet)
    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        accelerator.print(f"load network weights from {args.network_weights}: {info}")

    print(f'\n step 5. optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f' step 6. dataloader')
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    print(f'\n step 7. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. training preparing')
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs / {args.max_train_steps}")
    if args.full_fp16:
        assert (args.mixed_precision == "fp16"), "full_fp16 requires mixed precision='fp16'"
        accelerator.print("enable full fp16 training.")
        network.to(weight_dtype)
    elif args.full_bf16:
        assert (args.mixed_precision == "bf16"), "full_bf16 requires mixed precision='bf16' / mixed_precision='bf16'"
        accelerator.print("enable full bf16 training.")
        network.to(weight_dtype)

    unet.requires_grad_(False)
    unet.to(dtype=weight_dtype)
    for t_enc in text_encoders:
        t_enc.requires_grad_(False)

    if train_unet and train_text_encoder:
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler)
        text_encoders = [text_encoder]
    elif train_unet:
        unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, network, optimizer,
                                                                                       train_dataloader, lr_scheduler)
        text_encoder.to(accelerator.device)
    elif train_text_encoder:
        text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, network, optimizer, train_dataloader, lr_scheduler)
        text_encoders = [text_encoder]
        unet.to(accelerator.device, dtype=weight_dtype)  # move to device because unet is not prepared by accelerator
    else:
        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer,
                                                                                 train_dataloader, lr_scheduler)
    text_encoders = transform_models_if_DDP(text_encoders)
    unet, network = transform_models_if_DDP([unet, network])
    if args.gradient_checkpointing:
        unet.train()
        for t_enc in text_encoders:
            t_enc.train()
            if train_text_encoder:
                t_enc.text_model.embeddings.requires_grad_(True)
        if not train_text_encoder:  # train U-Net only
            unet.parameters().__next__().requires_grad_(True)
    else:
        unet.eval()
        for t_enc in text_encoders:
            t_enc.eval()
    del t_enc

    network.prepare_grad_etc(text_encoder, unet)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device, dtype=vae_dtype)

    print(f'\n step 8. training')
    args.save_every_n_epochs = 1
    attention_storer = AttentionStore()
    register_attention_control(unet, attention_storer)
    max_train_steps = len(train_dataloader) * args.max_train_epochs
    progress_bar = tqdm(range(max_train_steps),
                        smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                    num_train_timesteps=1000, clip_sample=False)
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    loss_list = []

    controller = AttentionStore()
    register_attention_control(unet, controller)

    for epoch in range(args.start_epoch, args.max_train_epochs):

        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")

        for step, batch in enumerate(train_dataloader):

            loss_dict = {}
            dist_loss = 0.0
            normal_cls_loss, normal_trigger_loss = 0.0, 0.0
            anormal_cls_loss, anormal_trigger_loss = 0.0, 0.0
            normal_feat_list, anormal_feat_list = [], []

            # -------------------------------------------- Holed Sample --------------------------------------------- #
            with torch.set_grad_enabled(True):
                input_ids = batch["input_ids"].to(accelerator.device)  # batch, 77 sen len
                enc_out = text_encoder(input_ids)  # batch, 77, 768
                encoder_hidden_states = enc_out["last_hidden_state"]

            with torch.no_grad():
                latents = vae.encode(batch["masked_image"].to(dtype=weight_dtype)).latent_dist.sample() # 1, 4, 64, 64
                latents = latents * vae_scale_factor  # [1,4,64,64]
            noise, noisy_latents, timesteps = get_noise_noisy_latents_partial_time(args, noise_scheduler,
                                           latents, min_timestep=args.min_timestep,max_timestep=args.max_timestep,)
            unet(noisy_latents, timesteps, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=None)

            object_mask = batch["object_mask"].to(accelerator.device)  # [1, 64, 64]
            object_mask = object_mask.flatten().squeeze()  # [64*64]
            mask = batch["masked_image_mask"].squeeze() # [64,64]
            mask_position = mask.flatten().squeeze()  # [64*64]
            background_mask = 1 - mask_position
            a_normal_mask = mask_position + background_mask
            normal_position = torch.where(a_normal_mask == 0, 1, 0)

            query_dict, attn_dict = controller.query_dict, controller.step_store
            controller.reset()
            for trg_layer in args.trg_layer_list:

                query = query_dict[trg_layer][0].squeeze(0)  # pix_num, dim
                pix_num = query.shape[0]

                for pix_idx in range(pix_num):
                    feat = query[pix_idx].squeeze(0)
                    flag = mask_position[pix_idx].item()
                    object_flag = object_mask[pix_idx].item()
                    if flag == 1:
                        anormal_feat_list.append(feat.unsqueeze(0))
                    elif object_flag == 1 :
                        normal_feat_list.append(feat.unsqueeze(0))

                attention_score = attn_dict[trg_layer][0]  # head, pix_num, 2
                cls_score, trigger_score = attention_score.chunk(2, dim=-1)
                cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # head, pix_num
                mask_position = mask_position.unsqueeze(0).repeat(cls_score.shape[0], 1)
                normal_cls_score = (cls_score *normal_position).mean(dim=0)
                normal_trigger_score = (trigger_score * normal_position).mean(dim=0)
                anormal_cls_score = (cls_score * mask_position).mean(dim=0)
                anormal_trigger_score = (trigger_score * mask_position).mean(dim=0)
                total_score = torch.ones_like(normal_cls_score)

                normal_cls_loss += (normal_cls_score / total_score)
                normal_trigger_loss += (1-(normal_trigger_score / total_score))
                anormal_cls_loss += (1-(anormal_cls_score / total_score))
                anormal_trigger_loss += (anormal_trigger_score / total_score)

            # ---------------------------------------- ANormal Sample Learning --------------------------------------- #
            with torch.no_grad():
                anomal_latents = vae.encode(batch['augmented_image'].to(dtype=weight_dtype)).latent_dist.sample()
                anomal_latents = anomal_latents * vae_scale_factor

            noise, anomal_noisy_latents, timesteps = get_noise_noisy_latents_partial_time(args, noise_scheduler,
                                                                                           anomal_latents)
            with accelerator.autocast():
                unet(anomal_noisy_latents, timesteps, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                     noise_type=None)

            query_dict, attn_dict = controller.query_dict, controller.step_store
            controller.reset()

            anomal_mask = batch['anomaly_mask'].squeeze() # [64,64]
            anormal_position = anomal_mask.flatten().squeeze() # [64*64]
            object_mask = batch["object_mask"].to(accelerator.device)  # [1, 64, 64]
            object_position = object_mask.flatten().squeeze()  # [64*64]
            background_mask = 1 - object_position
            a_normal_mask = object_position + background_mask
            normal_position = torch.where(a_normal_mask == 0, 1, 0)

            for trg_layer in args.trg_layer_list:
                query = query_dict[trg_layer][0].squeeze(0) # pix_num, dim
                pix_num = query.shape[0]
                for pix_idx in range(pix_num):
                    feat = query[pix_idx].squeeze(0)
                    anomal_flag = anormal_position[pix_idx].item()
                    object_flag = object_position[pix_idx].item()
                    if anomal_flag == 1 :
                        anormal_feat_list.append(feat.unsqueeze(0))
                    elif object_flag == 1 :
                        normal_feat_list.append(feat.unsqueeze(0))
                # ----------------------------------------- 3. attn loss --------------------------------------------- #
                attention_score = attn_dict[trg_layer][0] # head, pix_num, 2
                cls_score, trigger_score = attention_score.chunk(2, dim=-1)
                cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # head, pix_num

                # (1) get position
                anormal_position = anormal_position.unsqueeze(0).repeat(cls_score.shape[0], 1)

                anormal_cls_score = (cls_score * anormal_position).mean(dim=0)
                anormal_trigger_score = (trigger_score * anormal_position).mean(dim=0)

                normal_cls_score = (cls_score * normal_position).mean(dim=0)
                normal_trigger_score = (trigger_score * (1 - anormal_position)).mean(dim=0)

                total_score = torch.ones_like(normal_cls_score)

                normal_cls_loss += (normal_cls_score / total_score)
                normal_trigger_loss += (1 - (normal_trigger_score / total_score))
                anormal_cls_loss += (1 - (anormal_cls_score / total_score))
                anormal_trigger_loss += (anormal_trigger_score / total_score)

            # ----------------------------------------------------------------------------------------------------------
            # Distance
            anormal_feats = torch.cat(anormal_feat_list, dim=0)
            normal_feats = torch.cat(normal_feat_list, dim=0)
            mu = torch.mean(normal_feats, dim=0)
            cov = torch.cov(normal_feats.transpose(0, 1))
            anormal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in anormal_feats]
            anormal_dist_mean = torch.tensor(anormal_mahalanobis_dists).mean()
            normal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in normal_feats]
            normal_dist_mean = torch.tensor(normal_mahalanobis_dists).mean()
            total_dist = normal_dist_mean + anormal_dist_mean
            normal_dist_loss = normal_dist_mean / total_dist
            normal_dist_loss = normal_dist_loss * args.dist_loss_weight
            dist_loss += normal_dist_loss.requires_grad_()

            # ----------------------------------------------------------------------------------------------------------
            # Attention
            attn_loss = (normal_trigger_loss + anormal_trigger_loss)
            if args.do_cls_train :
                attn_loss += (normal_cls_loss + anormal_cls_loss)
            loss_dict['attn_loss'] = attn_loss.mean().item()

            # ----------------------------------------------------------------------------------------------------------
            # total loss
            loss = attn_loss.mean()
            if args.do_dist_loss:
                loss += dist_loss.mean()
                loss_dict['dist_loss'] = dist_loss.item()
            current_loss = loss.detach().item()
            if epoch == args.start_epoch :
                loss_list.append(current_loss)
            else:
                epoch_loss_total -= loss_list[step]
                loss_list[step] = current_loss
            epoch_loss_total += current_loss
            avr_loss = epoch_loss_total / len(loss_list)
            loss_dict['avr_loss'] = avr_loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                controller.reset()
            if is_main_process:
                progress_bar.set_postfix(**loss_dict)
            if global_step >= args.max_train_steps:
                break
        # ----------------------------------------------- Epoch Final ----------------------------------------------- #
        accelerator.wait_for_everyone()
        if args.save_every_n_epochs is not None:
            saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                        epoch + 1) < args.start_epoch + args.max_train_epochs
            if is_main_process and saving:
                ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                save_model(args, ckpt_name, accelerator.unwrap_model(network), save_dtype)
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--wandb_api_key', type=str, default='output')
    parser.add_argument('--wandb_project_name', type=str, default='bagel')
    # step 2. dataset
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument('--anomaly_source_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_repeat', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument('--perlin_max_scale', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    # step 3. preparing accelerator')
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--full_fp16", action="store_true",
                        help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients")
    # step 4. model
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='facebook/diffusion-dalle')
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1)")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / text encoder", )
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_dim", type=int, default=64,help="network dimensions (depends on each network) ")
    parser.add_argument("--network_alpha", type=float, default=4, help="alpha for LoRA weight scaling, default 1 ", )
    parser.add_argument("--network_dropout", type=float, default=None,
                        help="Drops neurons out of training every step", )
    parser.add_argument("--network_args", type=str, default=None, nargs="*",
                        help="additional argmuments for network (key=value)")
    parser.add_argument("--dim_from_weights", action="store_true",
                        help="automatically determine dim (rank) from network_weights / dim ", )
    parser.add_argument("--scale_weight_norms", type=float, default=None,
                        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. ", )
    parser.add_argument("--base_weights", type=str, default=None, nargs="*",
                        help="network weights to merge into the model before training", )
    parser.add_argument("--base_weights_multiplier", type=float, default=None, nargs="*",
                        help="multiplier for network weights to merge into the model before training ", )
    # step 5. optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                help="AdamW , AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov, "
              "SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, "
                             "DAdaptLion, DAdaptSGD, AdaFactor", )
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="use 8bit AdamW optimizer (requires bitsandbytes)", )
    parser.add_argument("--use_lion_optimizer", action="store_true",
                        help="use Lion optimizer (requires lion-pytorch)", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*",
                        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") ', )
    # lr
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module")
    parser.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*",
                        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）', )
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", help="scheduler to use for lr")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler (default is 0)", )
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1,
                        help="Number of restarts for cosine scheduler with restarts / cosine with restarts", )
    parser.add_argument("--lr_scheduler_power", type=float, default=1,
                        help="Polynomial power for polynomial scheduler / polynomial", )
    parser.add_argument('--text_encoder_lr', type=float, default=1e-5)
    parser.add_argument('--unet_lr', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_unet', action='store_true')
    parser.add_argument('--train_text_encoder', action='store_true')
    # training
    parser.add_argument("--total_normal_thred", type=float, default=0.5)
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--sample_every_n_steps", type=int, default=None,
                        help="generate sample images every N steps ")
    parser.add_argument("--sample_every_n_epochs", type=int, default=None,
                        help="generate sample images every N epochs (overwrites n_steps)", )
    parser.add_argument("--output_name", type=str, default=None, help="base name of trained model file ")
    parser.add_argument("--save_model_as", type=str, default="safetensors",
                        choices=[None, "ckpt", "pt", "safetensors"],
                        help="format to save the model (default is .safetensors)", )
    parser.add_argument("--training_comment", type=str, default=None,
                        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--do_dist_loss", action='store_true')
    parser.add_argument("--dist_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument("--anormal_weight", type=float, default=1.0)
    parser.add_argument('--normal_weight', type=float, default=1.0)
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    # step 7. inference check
    parser.add_argument("--scheduler_linear_start", type=float, default=0.00085)
    parser.add_argument("--scheduler_linear_end", type=float, default=0.012)
    parser.add_argument("--sample_sampler", type=str, default="ddim",  choices=["ddim", "pndm", "lms", "euler",
                                             "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++",
                                        "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a", ], )
    parser.add_argument("--scheduler_schedule", type=str, default="scaled_linear",
                                               choices=["scaled_linear", "linear", "cosine", "cosine_warmup", ], )
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--unet_inchannels", type=int, default=9)
    parser.add_argument("--back_token_separating", action='store_true')
    parser.add_argument("--min_timestep", type=int, default=0)
    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--down_dim", type=int)
    parser.add_argument("--noise_type", type=str)
    parser.add_argument("--truncating", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--negative_prompt", type=str,
                             default="low quality, worst quality, bad anatomy, bad composition, poor, low effort")
    parser.add_argument("--anomal_src_more", action = 'store_true')
    parser.add_argument("--without_background", action='store_true')
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    main(args)