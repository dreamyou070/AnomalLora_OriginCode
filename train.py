import argparse, math, random, json
from tqdm import tqdm
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
import torch
import os
from data.mvtec_sy import MVTecDRAEMTrainDataset
from model import load_target_model, transform_models_if_DDP
from model import create_network
from attention_store import AttentionStore
from model.tokenizer import load_tokenizer
from utils import get_epoch_ckpt_name, save_model, prepare_dtype
from utils.accelerator_utils import prepare_accelerator
from utils.attention_control import register_attention_control
from utils.optimizer_utils import get_optimizer, get_scheduler_fix
from utils.model_utils import prepare_scheduler_for_custom_training, get_noise_noisy_latents_and_timesteps
from model import unet_passing_argument
from utils.attention_control import passing_argument
from model import PositionalEmbedding
from utils import arg_as_list
from utils.utils_mahalanobis import gen_mahal_loss
from utils.model_utils import pe_model_save
#from utils.utils_loss import gen_attn_loss, FocalLoss
from utils.utils_loss import FocalLoss
from random import sample

def gen_value_dict(value_dict,
                   normal_cls_loss, anormal_cls_loss,
                   normal_trigger_loss, anormal_trigger_loss):
    if normal_cls_loss is not None:
        if 'normal_cls_loss' not in value_dict.keys():
            value_dict['normal_cls_loss'] = []
        value_dict['normal_cls_loss'].append(normal_cls_loss)
    if anormal_cls_loss is not None:
        if 'anormal_cls_loss' not in value_dict.keys():
            value_dict['anormal_cls_loss'] = []
        value_dict['anormal_cls_loss'].append(anormal_cls_loss)
    if normal_trigger_loss is not None:
        if 'normal_trigger_loss' not in value_dict.keys():
            value_dict['normal_trigger_loss'] = []
        value_dict['normal_trigger_loss'].append(normal_trigger_loss)
    if anormal_trigger_loss is not None:
        if 'anormal_trigger_loss' not in value_dict.keys():
            value_dict['anormal_trigger_loss'] = []
        value_dict['anormal_trigger_loss'].append(anormal_trigger_loss)
    return value_dict


def generate_attn_loss(target, position, base, do_lowering):
    target = target * position
    target_score = target / base
    if do_lowering:
        loss = target_score ** 1
    else:
        loss = (1 - target_score) ** 2
    return loss

def gen_attn_loss(value_dict):
    normal_cls_loss = torch.stack(value_dict['normal_cls_loss'], dim=0).mean(dim=0)
    anormal_cls_loss = torch.stack(value_dict['anormal_cls_loss'], dim=0).mean(dim=0)
    normal_trigger_loss = torch.stack(value_dict['normal_trigger_loss'], dim=0).mean(dim=0)
    anormal_trigger_loss = torch.stack(value_dict['anormal_trigger_loss'], dim=0).mean(dim=0)
    return normal_cls_loss, normal_trigger_loss, anormal_cls_loss, anormal_trigger_loss

def main(args):

    print(f'\n step 1. setting')
    output_dir = args.output_dir
    print(f' *** output_dir : {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    args.logging_dir = os.path.join(output_dir, 'log')
    os.makedirs(args.logging_dir, exist_ok=True)
    logging_file = os.path.join(args.logging_dir, 'log_8.txt')
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f'\n step 2. dataset and dataloader')
    if args.seed is None: args.seed = random.randint(0, 2 ** 32)
    set_seed(args.seed)
    tokenizer = load_tokenizer(args)
    if args.use_small_anomal :
        args.anomal_source_path = os.path.join(args.data_path, f"anomal_source_{args.obj_name}")
    dataset = MVTecDRAEMTrainDataset(root_dir=os.path.join(args.data_path, f'{args.obj_name}/train/good/rgb'),
                                     anomaly_source_path=args.anomal_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption=args.trigger_word,
                                     use_perlin=True,
                                     anomal_only_on_object=args.anomal_only_on_object,
                                     anomal_training=True,
                                     latent_res=args.latent_res,
                                     perlin_max_scale=args.perlin_max_scale,
                                     kernel_size=args.kernel_size,
                                     beta_scale_factor=args.beta_scale_factor,
                                     use_sharpen_aug=args.use_sharpen_aug,
                                     do_anomal_hole = args.do_anomal_hole)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

    print(' (4.3) positional embedding model')
    position_embedder = None
    if args.use_position_embedder:
        position_embedder = PositionalEmbedding(max_len=args.latent_res * args.latent_res, d_model=args.d_dim)

    print(f'\n step 5. optimizer')
    trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
    if position_embedder is not None:
        trainable_params.append({"params": position_embedder.parameters(),
                                 "lr": args.learning_rate})
    optimizer_name, optimizer_args, optimizer = get_optimizer(args, trainable_params)

    print(f'\n step 6. lr')
    lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

    print(f'\n step 7. loss function')
    loss_focal = FocalLoss()
    loss_l2 = torch.nn.modules.loss.MSELoss(reduction='none')

    print(f'\n step 8. weight dtype and network to accelerate preparing')
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
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler, position_embedder = accelerator.prepare(
            unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler, position_embedder)
        text_encoders = [text_encoder]
    text_encoders = transform_models_if_DDP(text_encoders)
    unet, network = transform_models_if_DDP([unet, network])
    if args.gradient_checkpointing:
        unet.train()
        position_embedder.train()
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

    print(f'\n step 8. Training !')
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs / {args.max_train_steps}")
    args.save_every_n_epochs = 1
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

    if is_main_process:
        if args.do_down_dim_mahal_loss:
            logging_info = f"'step', 'normal dist max', 'down dimed normal dist max'"
        else:
            logging_info = f"'step', 'normal dist max'"
        with open(logging_file, 'a') as f:
            f.write(logging_info + '\n')

    mu, cov = None, None
    for epoch in range(args.start_epoch, args.max_train_epochs):
        epoch_loss_total = 0
        accelerator.print(f"\nepoch {epoch + 1}/{args.start_epoch + args.max_train_epochs}")

        for step, batch in enumerate(train_dataloader):

            loss = torch.tensor(0.0, dtype=weight_dtype, device=accelerator.device)
            task_loss, dist_loss, attn_loss, map_loss = 0.0, 0.0, 0.0, 0.0
            normal_feat_list, anormal_feat_list = [], []
            value_dict, loss_dict = {}, {}

            if args.do_down_dim_mahal_loss :
                original_dim = 320
                down_dim_idx = torch.tensor(sample(range(0, original_dim), args.down_dim))
                down_dim_normal_feat_list, down_dim_anormal_feat_list = [], []

            with torch.set_grad_enabled(True):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))["last_hidden_state"]

            # --------------------------------------------------------------------------------------------------------- #
            # [1] normal sample
            with torch.no_grad():
                latents = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor
            noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                              noise_type=position_embedder).sample
            target = noise
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])
            task_loss += loss.mean()  * args.task_loss_weight



            query_dict, attn_dict = controller.query_dict, controller.step_store
            controller.reset()
            for trg_layer in args.trg_layer_list:
                query = query_dict[trg_layer][0].squeeze(0)  # pix_num, dim
                pix_num = query.shape[0]
                for pix_idx in range(pix_num):
                    feat = query[pix_idx].squeeze(0)
                    normal_feat_list.append(feat.unsqueeze(0))
                    if args.do_down_dim_mahal_loss :
                        down_dim_feat = torch.index_select(feat, 0, down_dim_idx.to(feat.device))
                        down_dim_normal_feat_list.append(down_dim_feat.unsqueeze(0))
                # (2) attn loss
                attn_score = attn_dict[trg_layer][0]  # head, pix_num, 2
                cls_score, trigger_score = attn_score.chunk(2, dim=-1)
                cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()     # head, pix_num
                cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0) # pix_num
                normal_cls_score, normal_trigger_score = cls_score, trigger_score
                total_score = torch.ones_like(cls_score)
                anomal_position = torch.zeros_like(cls_score)
                normal_trigger_loss = generate_attn_loss(normal_trigger_score, anomal_position, total_score, do_lowering=False)
                normal_cls_loss = generate_attn_loss(normal_cls_score, anomal_position, total_score, do_lowering=True)
                value_dict = gen_value_dict(value_dict, normal_cls_loss, None, normal_trigger_loss, None)
                # (3)
                if not args.use_focal_loss:
                    normal_map = trigger_score.unsqueeze(0).view(int(math.sqrt(pix_num)), int(math.sqrt(pix_num)))
                    trg_normal_map = torch.ones_like(normal_map)  # [64,64]
                    l2_loss = loss_l2(normal_map.float(), trg_normal_map.float())
                    map_loss += l2_loss
                if args.use_focal_loss:
                    attn_score = attn_score.mean(dim=0)  # 64*64, 2
                    res = int(attn_score.shape[0] ** 0.5)
                    cls, trigger = torch.chunk(attn_score, 2, dim=1)
                    cls_map = cls.view(res, res).unsqueeze(0).unsqueeze(0)
                    trigger_map = trigger.view(res, res).unsqueeze(0).unsqueeze(0)
                    focal_loss_in = torch.cat([cls_map, trigger_map], 1)
                    focal_loss_trg = torch.zeros_like(focal_loss_in)[:,0,:,:]
                    if args.adv_focal_loss:
                        focal_loss_trg = 1-focal_loss_trg
                    focal_loss = loss_focal(focal_loss_in, focal_loss_trg.to(dtype=weight_dtype))
                    map_loss += focal_loss
            # --------------------------------------------------------------------------------------------------------- #
            # [2] Masked Sample Learning
            with torch.no_grad():
                latents = vae.encode(batch["augmented_image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor
            noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                 noise_type=position_embedder).sample

            anomal_map = batch["anomaly_mask"].squeeze().flatten().squeeze()  # [64*64]
            trg_map = (1-anomal_map.view(1, 1, args.latent_res, args.latent_res))
            loss = torch.nn.functional.mse_loss((noise_pred * trg_map).float(),
                                                (target*trg_map).float(), reduction="none")
            loss = loss.mean([1, 2, 3])
            task_loss += loss.mean() * args.task_loss_weight
            query_dict, attn_dict = controller.query_dict, controller.step_store
            controller.reset()
            for trg_layer in args.trg_layer_list:
                anomal_position = anomal_map.squeeze(0)  # [64*64]
                query = query_dict[trg_layer][0].squeeze(0)  # pix_num, dim
                for pix_idx in range(query.shape[0]):
                    feat = query[pix_idx].squeeze(0)
                    anomal_flag = anomal_position[pix_idx].item()
                    if args.do_down_dim_mahal_loss:
                        down_dim_feat = torch.index_select(feat, 0, down_dim_idx.to(feat.device))
                        down_dim_normal_feat_list.append(down_dim_feat.unsqueeze(0))
                    if anomal_flag != 0:
                        anormal_feat_list.append(feat.unsqueeze(0))
                        if args.do_down_dim_mahal_loss:
                            down_dim_anormal_feat_list.append(down_dim_feat.unsqueeze(0))
                    else:
                        normal_feat_list.append(feat.unsqueeze(0))
                        if args.do_down_dim_mahal_loss:
                            down_dim_normal_feat_list.append(down_dim_feat.unsqueeze(0))
                # [2] attn score
                attn_score = attn_dict[trg_layer][0]  # head, pix_num, 2
                cls_score, trigger_score = attn_score.chunk(2, dim=-1)
                cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # head, pix_num
                cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
                total_score = torch.ones_like(cls_score)
                anormal_cls_loss = generate_attn_loss(cls_score, anomal_position, total_score, do_lowering=False)
                anormal_trigger_loss = generate_attn_loss(trigger_score, anomal_position, total_score, do_lowering=True)
                normal_cls_loss = generate_attn_loss(cls_score, 1 - anomal_position, total_score, do_lowering=True)
                normal_trigger_loss = generate_attn_loss(trigger_score, 1 - anomal_position, total_score,
                                                         do_lowering=False)
                value_dict = gen_value_dict(value_dict, normal_cls_loss, anormal_cls_loss, normal_trigger_loss,
                                            anormal_trigger_loss)
                # [3] normal map
                if not args.use_focal_loss:
                    normal_map = trigger_score.unsqueeze(0).view(int(math.sqrt(pix_num)), int(math.sqrt(pix_num)))
                    trg_normal_map = (1 - anomal_map).view(int(math.sqrt(pix_num)), int(math.sqrt(pix_num)))
                    l2_loss = loss_l2(normal_map.float(), trg_normal_map.float())
                    map_loss += l2_loss
                if args.use_focal_loss:
                    attn_score = attn_score.mean(dim=0)  # 64*64, 2
                    res = int(attn_score.shape[0] ** 0.5)
                    cls, trigger = torch.chunk(attn_score, 2, dim=1)
                    cls_map = cls.view(res, res).unsqueeze(0).unsqueeze(0)
                    trigger_map = trigger.view(res, res).unsqueeze(0).unsqueeze(0)
                    focal_loss_in = torch.cat([cls_map, trigger_map], 1)
                    focal_loss_trg = anomal_map.view(1, 1, int(pix_num ** 0.5),int(pix_num ** 0.5))
                    if args.adv_focal_loss: # normal map
                        focal_loss_trg = 1-focal_loss_trg
                    focal_loss = loss_focal(focal_loss_in,focal_loss_trg.to(dtype=weight_dtype))
                    map_loss += focal_loss

            # [3] Masked Sample Learning
            if args.do_anomal_hole :
                with torch.no_grad():
                    latents = vae.encode(batch["masked_image"].to(dtype=weight_dtype)).latent_dist.sample() * args.vae_scale_factor  # [1,4,64,64]
                noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
                                                                                            noise_type=position_embedder).sample
                anomal_map = batch["masked_image_mask"].squeeze().flatten().squeeze()  # [64*64]
                anomal_map = torch.where(anomal_map > 0, 1, 0).to(accelerator.device).unsqueeze(0) # [1, 64*64]
                query_dict, attn_dict = controller.query_dict, controller.step_store
                controller.reset()
                for trg_layer in args.trg_layer_list:
                    anomal_position = anomal_map.squeeze(0)      # [64*64]
                    query = query_dict[trg_layer][0].squeeze(0)  # pix_num, dim
                    for pix_idx in range(query.shape[0]):
                        feat = query[pix_idx].squeeze(0)
                        anomal_flag = anomal_position[pix_idx].item()
                        if args.do_down_dim_mahal_loss :
                            down_dim_feat = torch.index_select(feat, 0, down_dim_idx.to(feat.device))
                            down_dim_normal_feat_list.append(down_dim_feat.unsqueeze(0))
                        if anomal_flag != 0 :
                            anormal_feat_list.append(feat.unsqueeze(0))
                            if args.do_down_dim_mahal_loss :
                                down_dim_anormal_feat_list.append(down_dim_feat.unsqueeze(0))
                        else :
                            normal_feat_list.append(feat.unsqueeze(0))
                            if args.do_down_dim_mahal_loss :
                                down_dim_normal_feat_list.append(down_dim_feat.unsqueeze(0))
                    # [2] attn score
                    attn_score = attn_dict[trg_layer][0]  # head, pix_num, 2
                    cls_score, trigger_score = attn_score.chunk(2, dim=-1)
                    cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()     # head, pix_num
                    cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0) # pix_num
                    total_score = torch.ones_like(cls_score)
                    anormal_cls_loss = generate_attn_loss(cls_score, anomal_position, total_score, do_lowering=False)
                    anormal_trigger_loss = generate_attn_loss(trigger_score, anomal_position, total_score, do_lowering=True)

                    normal_cls_loss = generate_attn_loss(cls_score, 1-anomal_position, total_score, do_lowering=True)
                    normal_trigger_loss = generate_attn_loss(trigger_score, 1-anomal_position, total_score, do_lowering=False)

                    value_dict = gen_value_dict(value_dict, normal_cls_loss, anormal_cls_loss, normal_trigger_loss,anormal_trigger_loss)
                    # [3] normal map
                    if not args.use_focal_loss:
                        normal_map = trigger_score.unsqueeze(0).view(int(math.sqrt(pix_num)), int(math.sqrt(pix_num)))
                        trg_normal_map = (1 - anomal_map).view(int(math.sqrt(pix_num)), int(math.sqrt(pix_num)))
                        l2_loss = loss_l2(normal_map.float(), trg_normal_map.float())
                        map_loss += l2_loss
                    if args.use_focal_loss:
                        attn_score = attn_score.mean(dim=0)  # 64*64, 2
                        res = int(attn_score.shape[0] ** 0.5)
                        cls, trigger = torch.chunk(attn_score, 2, dim=1)
                        cls_map = cls.view(res, res).unsqueeze(0).unsqueeze(0)
                        trigger_map = trigger.view(res, res).unsqueeze(0).unsqueeze(0)
                        focal_loss_in = torch.cat([cls_map, trigger_map], 1)
                        focal_loss_trg = anomal_map.view(1, 1, int(pix_num ** 0.5), int(pix_num ** 0.5))
                        if args.adv_focal_loss:  # normal map
                            focal_loss_trg = 1 - focal_loss_trg
                        focal_loss = loss_focal(focal_loss_in, focal_loss_trg.to(dtype=weight_dtype))
                        map_loss += focal_loss

            # querry permute
            queries = torch.cat(normal_feat_list, dim=0) #
            p_shuffle = torch.randperm(queries.size(0))
            shuffled_queries = queries[p_shuffle]
            normal_query_loss = loss_l2(queries.float(), shuffled_queries.float()).to(weight_dtype).mean()

            # ----------------------------------------------------------------------------------------------------------
            # [5] backprop
            if args.do_task_loss:
                loss += task_loss.to(weight_dtype)
                loss_dict['task_loss'] = task_loss.item()

            if args.do_dist_loss:
                normal_dist_max, normal_dist_loss, mu, cov = gen_mahal_loss(args, anormal_feat_list, normal_feat_list, mu, cov)
                dist_loss += normal_dist_loss.to(weight_dtype).requires_grad_()
                if args.do_down_dim_mahal_loss:
                    down_dim_normal_dist_max, down_dim_normal_dist_loss = gen_mahal_loss(args, down_dim_anormal_feat_list, down_dim_normal_feat_list)
                    down_dim_normal_dist_loss = down_dim_normal_dist_loss.to(weight_dtype)
                    dist_loss += down_dim_normal_dist_loss.requires_grad_()
                loss += dist_loss.to(weight_dtype)
                loss_dict['dist_loss'] = dist_loss.item()

            if args.do_attn_loss:
                normal_cls_loss, normal_trigger_loss, anormal_cls_loss, anormal_trigger_loss = gen_attn_loss(value_dict)
                attn_loss += args.normal_weight * normal_trigger_loss.mean() + args.anormal_weight * anormal_trigger_loss.mean()
                if args.do_cls_train:
                    attn_loss += args.normal_weight * normal_cls_loss.mean() + args.anormal_weight * anormal_cls_loss.mean()
                loss += attn_loss.mean().to(weight_dtype)
                loss_dict['attn_loss'] = attn_loss.mean().item()

            if args.do_map_loss:
                loss += map_loss.mean().to(weight_dtype)
                loss_dict['map_loss'] = map_loss.mean().item()

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
                if args.do_down_dim_mahal_loss:
                    logging_info = f'{global_step}, {normal_dist_max}, {down_dim_normal_dist_loss}'
                else:
                    logging_info = f'{global_step}, {normal_dist_max}'
                with open(logging_file, 'a') as f:
                    f.write(logging_info + '\n')
                progress_bar.set_postfix(**loss_dict)
            if global_step >= args.max_train_steps:
                break
        # ----------------------------------------------------------------------------------------------------------- #
        # [6] epoch final
        accelerator.wait_for_everyone()
        if args.save_every_n_epochs is not None:
            saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                    epoch + 1) < args.start_epoch + args.max_train_epochs
            if is_main_process and saving:
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
    parser.add_argument('--use_sharpen_aug', action='store_true')

    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument("--anomal_source_path", type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument('--perlin_max_scale', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    # step 3. preparing accelerator
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients")
    parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients")
    parser.add_argument("--lowram", action="store_true", )
    parser.add_argument("--no_half_vae", action="store_true",
                        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precision", )
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
    parser.add_argument("--use_8bit_adam", action="store_true", help="use 8bit AdamW optimizer(requires bitsandbytes)",)
    parser.add_argument("--use_lion_optimizer", action="store_true", help="use Lion optimizer (requires lion-pytorch)",)
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

    # step 8. training
    parser.add_argument("--output_name", type=str, default=None, help="base name of trained model file ")
    parser.add_argument("--save_model_as", type=str, default="safetensors",
               choices=[None, "ckpt", "pt", "safetensors"], help="format to save the model (default is .safetensors)", )
    parser.add_argument("--training_comment", type=str, default=None,
                         help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列")
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
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    # step 7. inference check
    parser.add_argument("--scheduler_linear_start", type=float, default=0.00085)
    parser.add_argument("--scheduler_linear_end", type=float, default=0.012)
    parser.add_argument("--scheduler_schedule", type=str, default="scaled_linear",
                        choices=["scaled_linear", "linear", "cosine", "cosine_warmup", ], )
    parser.add_argument("--down_dim", type=int)
    parser.add_argument("--noise_type", type=str)
    parser.add_argument("--anomal_src_more", action='store_true')
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    parser.add_argument("--beta_scale_factor", type=float, default=0.4)
    parser.add_argument("--do_map_loss", action='store_true')
    parser.add_argument("--image_classification_layer", type=str, )
    parser.add_argument("--use_small_anomal", action='store_true')
    parser.add_argument("--do_anomal_hole", action='store_true')
    parser.add_argument("--do_down_dim_mahal_loss", action='store_true')
    parser.add_argument("--use_focal_loss", action='store_true')
    # ---------------------------------------------------------------------------------------------------------------- #
    parser.add_argument("--num_ddim_steps", type=int, default=30)
    parser.add_argument("--do_local_self_attn", action='store_true')
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--only_local_self_attn", action='store_true')
    parser.add_argument("--fixed_window_size", action='store_true')
    parser.add_argument("--do_add_query", action='store_true')
    parser.add_argument("--add_query_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--sample_every_n_steps", type=int, default=None, help="generate sample images every N steps ")
    parser.add_argument("--sample_every_n_epochs", type=int, default=None,
                        help="generate sample images every N epochs (overwrites n_steps)", )
    parser.add_argument("--adv_focal_loss", action='store_true')
    parser.add_argument("--previous_mahal", action='store_true')
    parser.add_argument("--do_task_loss", action='store_true')
    args = parser.parse_args()
    unet_passing_argument(args)
    passing_argument(args)
    main(args)
