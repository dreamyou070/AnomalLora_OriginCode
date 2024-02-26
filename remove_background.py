import os
import argparse, torch
from model.lora import LoRANetwork,LoRAInfModule
from attention_store import AttentionStore
from utils.attention_control import passing_argument
from model.unet import unet_passing_argument
from utils.attention_control import register_attention_control
from accelerate import Accelerator
from model.tokenizer import load_tokenizer
from utils import prepare_dtype
from utils.model_utils import get_input_ids
from PIL import Image
from utils.image_utils import load_image, image2latent
import numpy as np
from model.diffusion_model import load_target_model
from model.pe import PositionalEmbedding
from safetensors.torch import load_file
from attention_store.normal_activator import NormalActivator
from attention_store.normal_activator import passing_normalize_argument

def inference(latent,
              tokenizer, text_encoder, unet, controller, normal_activator, position_embedder,
              args, org_h, org_w, thred):
    # [1] text
    input_ids, attention_mask = get_input_ids(tokenizer, args.prompt)
    encoder_hidden_states = text_encoder(input_ids.to(text_encoder.device))["last_hidden_state"]
    # [2] unet
    unet(latent, 0, encoder_hidden_states, trg_layer_list=args.trg_layer_list,
         noise_type=position_embedder)
    query_dict, attn_dict = controller.query_dict, controller.step_store
    controller.reset()
    if args.single_layer:
        for trg_layer in args.trg_layer_list:
            attn_score = attn_dict[trg_layer][0]  # head, pix_num, 2
    else:
        for trg_layer in args.trg_layer_list:
            normal_activator.resize_attn_scores(attn_dict[trg_layer][0])
        attn_score = normal_activator.generate_conjugated_attn_score()
    cls_map = attn_score[:, :, 0].squeeze().mean(dim=0)  # [res*res]
    trigger_map = attn_score[:, :, 1].squeeze().mean(dim=0)
    pix_num = trigger_map.shape[0]
    res = int(pix_num ** 0.5)
    cls_map = cls_map.unsqueeze(0).view(res, res)
    cls_map_pil = Image.fromarray((255 * cls_map).cpu().detach().numpy().astype(np.uint8)).resize((org_h, org_w))
    normal_map = torch.where(trigger_map > thred, 1, trigger_map).squeeze()
    normal_map = normal_map.unsqueeze(0).view(res, res)
    normal_map_pil = Image.fromarray(
        normal_map.cpu().detach().numpy().astype(np.uint8) * 255).resize((org_h, org_w))
    anomal_np = ((1 - normal_map) * 255).cpu().detach().numpy().astype(np.uint8)
    anomaly_map_pil = Image.fromarray(anomal_np).resize((org_h, org_w))

    return cls_map_pil, normal_map_pil, anomaly_map_pil


def generate_object_point(object_mask_pil):
    object_mask_np = np.array(object_mask_pil)
    h, w = object_mask_np.shape
    h_indexs, w_indexs = [], []
    for h_i in range(h):
        for w_i in range(w):
            if object_mask_np[h_i, w_i] > 0:
                h_indexs.append(h_i)
                w_indexs.append(w_i)

    h_start, h_end = min(h_indexs), max(h_indexs)
    w_start, w_end = min(w_indexs), max(w_indexs)

    h_pad = 0.02 * h
    w_pad = 0.02 * w
    h_start = h_start - h_pad if h_start - h_pad > 0 else 0
    h_end = h_end + h_pad if h_end + h_pad < h else h
    w_start = w_start - w_pad if w_start - w_pad > 0 else 0
    w_end = w_end + w_pad if w_end + w_pad < w else w
    h_start, h_end, w_start, w_end = int(h_start), int(h_end), int(w_start), int(w_end)

    return h_start, h_end, w_start, w_end

def main(args):

    print(f'\n step 1. accelerator')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.log_with,
                              project_dir='log')

    print(f'\n step 2. model')
    weight_dtype, save_dtype = prepare_dtype(args)
    tokenizer = load_tokenizer(args)
    text_encoder, vae, unet, _ = load_target_model(args, weight_dtype,
                                                   accelerator)

    if args.use_position_embedder:
        position_embedder = PositionalEmbedding(max_len=args.latent_res * args.latent_res,
                                                d_model=args.d_dim)

    print(f'\n step 2. accelerator and device')
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    print(f'\n step 3. inference')
    models = os.listdir(args.network_folder)
    network = LoRANetwork(text_encoder=text_encoder,
                          unet=unet,
                          lora_dim=args.network_dim,
                          alpha=args.network_alpha,
                          module_class=LoRAInfModule)
    network.apply_to(text_encoder, unet, True, True)
    raw_state_dict = network.state_dict()
    raw_state_dict_orig = raw_state_dict.copy()

    normal_activator = NormalActivator(None, None, args.use_focal_loss)

    for model in models:

        network_model_dir = os.path.join(args.network_folder, model)
        lora_name, ext = os.path.splitext(model)
        lora_epoch = int(lora_name.split('-')[-1])

        # [1] loead pe
        parent = os.path.split(args.network_folder)[0]
        pe_base_dir = os.path.join(parent, f'position_embedder')
        pretrained_pe_dir = os.path.join(pe_base_dir, f'position_embedder_{lora_epoch}.safetensors')
        position_embedder_state_dict = load_file(pretrained_pe_dir)
        position_embedder.load_state_dict(position_embedder_state_dict)
        position_embedder.to(accelerator.device, dtype=weight_dtype)

        # [2] load network
        anomal_detecting_state_dict = load_file(network_model_dir)
        for k in anomal_detecting_state_dict.keys():
            raw_state_dict[k] = anomal_detecting_state_dict[k]
        network.load_state_dict(raw_state_dict)
        network.to(accelerator.device, dtype=weight_dtype)

        # [3] files
        parent, _ = os.path.split(args.network_folder)
        recon_base_folder = os.path.join(parent, 'reconstruction_background_removing')
        os.makedirs(recon_base_folder, exist_ok=True)
        lora_base_folder = os.path.join(recon_base_folder, f'lora_epoch_{lora_epoch}')
        os.makedirs(lora_base_folder, exist_ok=True)

        # [4] collector
        controller = AttentionStore()
        register_attention_control(unet, controller)

        for thred in args.threds :
            thred_folder = os.path.join(lora_base_folder, f'thred_{thred}')
            os.makedirs(thred_folder, exist_ok=True)

            check_base_folder = os.path.join(thred_folder, f'my_check')
            os.makedirs(check_base_folder, exist_ok=True)
            answer_base_folder = os.path.join(thred_folder, f'scoring/{args.obj_name}/test')
            os.makedirs(answer_base_folder, exist_ok=True)

            # [1] test path
            test_img_folder = args.data_path
            parent, test_folder = os.path.split(test_img_folder)

            anomal_folders = os.listdir(test_img_folder)
            for anomal_folder in anomal_folders:
                answer_anomal_folder = os.path.join(answer_base_folder, anomal_folder)
                os.makedirs(answer_anomal_folder, exist_ok=True)
                save_base_folder = os.path.join(check_base_folder, anomal_folder)
                os.makedirs(save_base_folder, exist_ok=True)


                anomal_folder_dir = os.path.join(test_img_folder, anomal_folder)
                rgb_folder = os.path.join(anomal_folder_dir, 'rgb')
                gt_folder = os.path.join(anomal_folder_dir, 'gt')
                if args.object_crop:
                    object_mask_folder = os.path.join(anomal_folder_dir, 'object_mask')
                rgb_imgs = os.listdir(rgb_folder)

                for rgb_img in rgb_imgs:

                    name, ext = os.path.splitext(rgb_img)
                    rgb_img_dir = os.path.join(rgb_folder, rgb_img)
                    pil_img = Image.open(rgb_img_dir).convert('RGB')
                    org_h, org_w = pil_img.size

                    # [1] read object mask
                    if args.object_crop :
                        object_mask_pil = Image.open(os.path.join(object_mask_folder, rgb_img)).convert('L')
                        h_start, h_end, w_start, w_end = generate_object_point(object_mask_pil)
                        input_img = pil_img.crop((w_start, h_start, w_end, h_end))
                    else :
                        input_img = pil_img
                    trg_h, trg_w = input_img.size
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            img = np.array(input_img.resize((512, 512)))
                            vae_latent = image2latent(img, vae, weight_dtype)
                            cls_map_pil, normal_map_pil, anomaly_map_pil = inference(vae_latent,
                                                                                     tokenizer, text_encoder, unet,
                                                                                     controller, normal_activator,
                                                                                     position_embedder,
                                                                                     args,
                                                                                     trg_h, trg_w,
                                                                                     thred)
                            # pillow img = normal_map_pil
                            normal_np = np.array(normal_map_pil) # [512,512]
                            normal_position = np.where(normal_np > 0, 1, 0)
                            normal_position = np.expand_dims(normal_position, axis=2).repeat(3, axis=2)
                            back_removed_img = np.array(input_img) * normal_position
                            back_removed_img_pil = Image.fromarray(back_removed_img).convert('RGB')
                            back_removed_img_pil.save(os.path.join(save_base_folder, f'{name}_back_removed{ext}'))
                    controller.reset()
                    normal_activator.reset()
            # ---------------------------------------------------------------------------------------------------------
            # [2] train path
            if not args.object_crop:
                train_img_folder = os.path.join(parent, 'train')

                save_base_folder = os.path.join(check_base_folder, f'train_good')
                os.makedirs(save_base_folder, exist_ok=True)

                normal_folder_dir = os.path.join(train_img_folder, 'good')
                rgb_folder = os.path.join(normal_folder_dir, 'rgb')

                if args.object_crop:
                    object_mask_folder = os.path.join(anomal_folder_dir, 'object_mask')

                rgb_imgs = os.listdir(rgb_folder)
                for rgb_img in rgb_imgs:

                    name, ext = os.path.splitext(rgb_img)
                    rgb_img_dir = os.path.join(rgb_folder, rgb_img)
                    pil_img = Image.open(rgb_img_dir).convert('RGB')
                    org_h, org_w = pil_img.size

                    # [1] read object mask
                    if args.object_crop :
                        object_mask_pil = Image.open(os.path.join(object_mask_folder, rgb_img)).convert('L')
                        h_start, h_end, w_start, w_end = generate_object_point(object_mask_pil)
                        input_img = pil_img.crop((w_start, h_start, w_end, h_end))
                    else :
                        input_img = pil_img
                    trg_h, trg_w = input_img.size
                    if accelerator.is_main_process:

                        with torch.no_grad():
                            img = np.array(input_img.resize((512, 512)))
                            vae_latent = image2latent(img, vae, weight_dtype)
                            cls_map_pil, normal_map_pil, anomaly_map_pil = inference(vae_latent,
                                                                                     tokenizer, text_encoder, unet,
                                                                                     controller, normal_activator,
                                                                                     position_embedder,
                                                                                     args,
                                                                                     trg_h, trg_w,
                                                                                     thred)
                            # pillow img = normal_map_pil
                            normal_np = np.array(normal_map_pil)  # [512,512]
                            normal_position = np.where(normal_np > 0, 1, 0)
                            normal_position = np.expand_dims(normal_position, axis=2).repeat(3, axis=2)
                            back_removed_img = np.array(input_img) * normal_position
                            back_removed_img_pil = Image.fromarray(back_removed_img).convert('RGB')
                            back_removed_img_pil.save(os.path.join(save_base_folder, f'{name}_back_removed{ext}'))
        print(f'Model To Original')
        for k in raw_state_dict_orig.keys():
            raw_state_dict[k] = raw_state_dict_orig[k]
        network.load_state_dict(raw_state_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomal Lora')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default='facebook/diffusion-dalle')
    parser.add_argument('--network_dim', type=int, default=64)
    parser.add_argument('--network_alpha', type=float, default=4)
    parser.add_argument('--network_folder', type=str)
    parser.add_argument("--lowram", action="store_true", )
    # step 4. dataset and dataloader
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument('--data_path', type=str,
                        default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='bagel')
    # step 6
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--prompt", type=str, default="bagel", )
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--single_layer", action='store_true')
    parser.add_argument("--use_noise_scheduler", action='store_true')
    parser.add_argument('--min_timestep', type=int, default=0)
    parser.add_argument('--max_timestep', type=int, default=500)
    # step 8. test
    import ast
    def arg_as_list(arg):
        v = ast.literal_eval(arg)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (arg))
        return v
    parser.add_argument("--threds", type=arg_as_list,default=[0.85,])
    parser.add_argument("--trg_layer_list", type=arg_as_list, default=[])
    parser.add_argument("--position_embedding_layer", type=str)
    parser.add_argument("--use_position_embedder", action='store_true')
    parser.add_argument("--do_normalized_score", action='store_true')
    parser.add_argument("--d_dim", default=320, type=int)
    parser.add_argument("--thred", default=0.5, type=float)
    parser.add_argument("--image_classification_layer", type=str)
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--gen_batchwise_attn", action='store_true')
    parser.add_argument("--object_crop", action='store_true')
    args = parser.parse_args()
    passing_argument(args)
    unet_passing_argument(args)
    passing_normalize_argument(args)
    main(args)