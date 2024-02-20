import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch
import os
from attention_store import AttentionStore
from attention_store.normal_activator import NormalActivator
from model.diffusion_model import transform_models_if_DDP
from model.unet import unet_passing_argument
from utils import get_epoch_ckpt_name, save_model, prepare_dtype, arg_as_list
from utils.attention_control import passing_argument, register_attention_control
from utils.accelerator_utils import prepare_accelerator
from utils.optimizer_utils import get_optimizer, get_scheduler_fix
from utils.model_utils import prepare_scheduler_for_custom_training, get_noise_noisy_latents_and_timesteps
from utils.model_utils import pe_model_save
from utils.utils_loss import FocalLoss
from data.prepare_dataset import call_dataset
from model import call_model_package


def main(args):

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    print(f' *** output_dir : {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(args.logging_dir, exist_ok=True)
    logging_file = os.path.join(args.logging_dir, 'log.txt')

    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'\n step 2. dataset and dataloader')
    if args.seed is None: args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    train_dataloader = call_dataset(args)

    print(f'\n step 3. preparing accelerator')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f'\n step 4. model ')
    weight_dtype, save_dtype = prepare_dtype(args)
    text_encoder, vae, unet, network, position_embedder = call_model_package(args, weight_dtype, accelerator)

    print(f'\n step 5. optimizer')
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    trainable_params.append({"params": position_embedder.parameters(), "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f'\n step 6. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. loss function')
    loss_focal = FocalLoss()
    loss_l2 = torch.nn.modules.loss.MSELoss(reduction='none')
    normal_activator = NormalActivator(loss_focal, loss_l2, args.use_focal_loss)

    print(f'\n step 8. model to device')
    unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler, position_embedder = accelerator.prepare(
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler, position_embedder)
    text_encoders = transform_models_if_DDP([text_encoder])
    unet, network = transform_models_if_DDP([unet, network])
    if args.gradient_checkpointing:
        unet.train()
        position_embedder.train()
        for t_enc in text_encoders:
            t_enc.train()
            if args.train_text_encoder:
                t_enc.text_model.embeddings.requires_grad_(True)
        if not args.train_text_encoder:  # train U-Net only
            unet.parameters().__next__().requires_grad_(True)
    else:
        unet.eval()
        for t_enc in text_encoders:
            t_enc.eval()
    del t_enc
    network.prepare_grad_etc(text_encoder, unet)
    vae.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 9. Training !')
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0,
                        disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0
    loss_list = []
    controller = AttentionStore()
    register_attention_control(unet, controller)
    if is_main_process:
        logging_info = f"'step', 'normal dist mean', 'normal dist max'"
        with open(logging_file, 'a') as f:
            f.write(logging_info + '\n')


    for epoch in range(args.start_epoch, args.max_train_epochs):
        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")

        for step, batch in enumerate(train_dataloader):

            device = accelerator.device
            loss = torch.tensor(0.0, dtype=weight_dtype, device=accelerator.device)
            loss_dict = {}
            with torch.set_grad_enabled(True):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))["last_hidden_state"]
            # --------------------------------------------------------------------------------------------------------- #
            if args.do_normal_sample:
                with torch.no_grad():
                    latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor
                anomal_position_vector = torch.zeros_like(batch['object_mask'].squeeze().flatten())
                with torch.set_grad_enabled(True):
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
                query_dict, attn_dict = controller.query_dict, controller.step_store
                controller.reset()
                for trg_layer in args.trg_layer_list:
                    query = query_dict[trg_layer][0].squeeze(0)  # pix_num, dim
                    normal_activator.resize_query_features(query)
                    attn_score = attn_dict[trg_layer][0]  # head, pix_num, 2
                    normal_activator.resize_attn_scores(attn_score)
                c_query = normal_activator.generate_conjugated()
                normal_activator.collect_queries(c_query, anomal_position_vector, do_collect_normal = True)
                # [2]
                c_attn_score = normal_activator.generate_conjugated_attn_score()
                normal_activator.collect_attention_scores(c_attn_score, anomal_position_vector)
                # [3]
                normal_activator.collect_anomal_map_loss(c_attn_score, anomal_position_vector)
            # --------------------------------------------------------------------------------------------------------- #
            if args.do_anomal_sample:
                with torch.no_grad():
                    latents = vae.encode(batch["anomal_image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor
                anomal_position_vector = batch["anomal_mask"].squeeze().flatten()
                with torch.set_grad_enabled(True):
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list, noise_type=position_embedder)
                query_dict, attn_dict = controller.query_dict, controller.step_store
                controller.reset()
                for trg_layer in args.trg_layer_list:
                    query = query_dict[trg_layer][0].squeeze(0)  # pix_num, dim
                    normal_activator.resize_query_features(query)
                    attn_score = attn_dict[trg_layer][0]  # head, pix_num, 2
                    normal_activator.resize_attn_scores(attn_score)
                c_query = normal_activator.generate_conjugated()
                normal_activator.collect_queries(c_query, anomal_position_vector)
                c_attn_score = normal_activator.generate_conjugated_attn_score()
                normal_activator.collect_attention_scores(c_attn_score, anomal_position_vector)
                normal_activator.collect_anomal_map_loss(c_attn_score, anomal_position_vector)
            # --------------------------------------------------------------------------------------------------------- #
            if args.do_background_masked_sample:
                with torch.no_grad():
                    latents = vae.encode(batch["bg_anomal_image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor
                anomal_position_vector = batch["bg_anomal_mask"].squeeze().flatten()
                with torch.set_grad_enabled(True):
                    unet(latents, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                         noise_type=position_embedder)
                query_dict, attn_dict = controller.query_dict, controller.step_store
                controller.reset()
                for trg_layer in args.trg_layer_list:
                    query = query_dict[trg_layer][0].squeeze(0)  # pix_num, dim
                    normal_activator.resize_query_features(query)
                    attn_score = attn_dict[trg_layer][0]  # head, pix_num, 2
                    normal_activator.resize_attn_scores(attn_score)
                c_query = normal_activator.generate_conjugated()
                normal_activator.collect_queries(c_query, anomal_position_vector)
                c_attn_score = normal_activator.generate_conjugated_attn_score()
                normal_activator.collect_attention_scores(c_attn_score, anomal_position_vector)
                normal_activator.collect_anomal_map_loss(c_attn_score, anomal_position_vector)
            # ----------------------------------------------------------------------------------------------------------
            # [5] backprop
            dist_loss, normal_dist_mean, normal_dist_max = normal_activator.generate_mahalanobis_distance_loss()
            if args.do_dist_loss:
                loss += dist_loss
                loss_dict['dist_loss'] = dist_loss.item()

            if args.do_attn_loss:
                normal_cls_loss, normal_trigger_loss, anormal_cls_loss, anormal_trigger_loss = normal_activator.generate_attention_loss()
                if type(anormal_cls_loss) == float:
                    attn_loss = args.normal_weight * normal_trigger_loss.mean()
                else:
                    attn_loss = args.normal_weight * normal_cls_loss.mean() + args.anormal_weight * anormal_cls_loss.mean()
                if args.do_cls_train:
                    if type(anormal_trigger_loss) == float:
                        attn_loss = args.normal_weight * normal_cls_loss.mean()
                    else:
                        attn_loss += args.normal_weight * normal_cls_loss.mean() + args.anormal_weight * anormal_cls_loss.mean()
                loss += attn_loss
                loss_dict['attn_loss'] = attn_loss.item()

            if args.do_map_loss:
                map_loss = normal_activator.generate_anomal_map_loss()
                loss += map_loss
                loss_dict['map_loss'] = map_loss.item()

            loss = loss.to(weight_dtype)
            current_loss = loss.detach().item()
            if epoch == args.start_epoch:
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
            if is_main_process:
                logging_info = f'{global_step}, {normal_dist_mean}, {normal_dist_max}'
                with open(logging_file, 'a') as f:
                    f.write(logging_info + '\n')
                progress_bar.set_postfix(**loss_dict)
            normal_activator.reset()
            controller.reset()
            if global_step >= args.max_train_steps:
                break
            # ----------------------------------------------------------------------------------------------------------- #
            # [6] epoch final
        accelerator.wait_for_everyone()
        if is_main_process:
            ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
            save_model(args, ckpt_name, accelerator.unwrap_model(network), save_dtype)
            if position_embedder is not None:
                position_embedder_base_save_dir = os.path.join(args.output_dir, 'position_embedder')
                os.makedirs(position_embedder_base_save_dir, exist_ok=True)
                p_save_dir = os.path.join(position_embedder_base_save_dir,
                                          f'position_embedder_{epoch + 1}.safetensors')
                pe_model_save(accelerator.unwrap_model(position_embedder), save_dtype, p_save_dir)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    # step 2. dataset
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument("--anomal_source_path", type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument('--perlin_max_scale', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--reference_check", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--use_small_anomal", action='store_true')
    parser.add_argument("--beta_scale_factor", type=float, default=0.8)
    parser.add_argument("--bgrm_test", action='store_true')
    # step 3. preparing accelerator
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--d_dim", default=320, type=int)
    # step 4. model
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='facebook/diffusion-dalle')
    parser.add_argument("--clip_skip", type=int, default=None,
                        help="use output of nth layer from back of text encoder (n>=1)")
    parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                        help="max token length of text encoder (default for 75, 150 or 225) / text encoder", )
    parser.add_argument("--vae_scale_factor", type=float, default=0.18215)
    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_dim", type=int, default=64, help="network dimensions (depends on each network) ")
    parser.add_argument("--network_alpha", type=float, default=4, help="alpha for LoRA weight scaling, default 1 ", )
    parser.add_argument("--network_dropout", type=float, default=None,)
    parser.add_argument("--network_args", type=str, default=None, nargs="*",)
    parser.add_argument("--dim_from_weights", action="store_true",)
    # step 5. optimizer
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
            help="AdamW , AdamW8bit, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov,"
            "SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP,"
             "DAdaptLion, DAdaptSGD, AdaFactor", )
    parser.add_argument("--use_8bit_adam", action="store_true", help="use 8bit AdamW optimizer(requires bitsandbytes)",)
    parser.add_argument("--use_lion_optimizer", action="store_true",
                        help="use Lion optimizer (requires lion-pytorch)", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping")
    parser.add_argument("--optimizer_args", type=str, default=None, nargs="*",
                        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") ', )
    # lr
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module")
    parser.add_argument("--lr_scheduler_args", type=str, default=None, nargs="*",
                        help='additional arguments for scheduler (like "T_max=100")')
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
    # step 8. training
    parser.add_argument("--save_model_as", type=str, default="safetensors",
              choices=[None, "ckpt", "pt", "safetensors"], help="format to save the model (default is .safetensors)",)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    # [1]
    parser.add_argument("--do_object_detection", action='store_true')
    parser.add_argument("--do_normal_sample", action='store_true')
    parser.add_argument("--do_anomal_sample", action='store_true')
    parser.add_argument("--do_background_masked_sample", action='store_true')
    # [2]
    parser.add_argument("--do_dist_loss", action='store_true')
    parser.add_argument("--dist_loss_weight", type=float, default=1.0)
    parser.add_argument("--do_attn_loss", action='store_true')
    parser.add_argument("--do_cls_train", action='store_true')
    parser.add_argument("--attn_loss_weight", type=float, default=1.0)
    parser.add_argument("--anormal_weight", type=float, default=1.0)
    parser.add_argument('--normal_weight', type=float, default=1.0)
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--do_map_loss", action='store_true')
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--use_noise_scheduler", action='store_true')
    parser.add_argument("--do_normalized_score", action='store_true')

    # [3]
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    from attention_store.normal_activator import passing_normalize_argument
    passing_normalize_argument(args)
    main(args)