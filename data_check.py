import argparse
import torch
from model.tokenizer import load_tokenizer
import numpy as np
import os
from PIL import Image
from data.mvtec import passing_mvtec_argument, MVTecDRAEMTrainDataset

def main(args):

    print(f'\n step 2. dataset')
    obj_name = args.obj_name
    root_dir = f'/home/dreamyou070/MyData/anomaly_detection/MVTec/{obj_name}/train_object_detector'
    Enum_images = len(os.listdir(root_dir))
    #print(f'num_images: {num_images}')
    args.anomaly_source_path = '/home/dreamyou070/MyData/anomal_source'
    dataset = MVTecDRAEMTrainDataset(root_dir=root_dir,
                                     anomaly_source_path=args.anomaly_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer = None,
                                     caption=obj_name,
                                     use_perlin=True,
                                     anomal_only_on_object=args.anomal_only_on_object,
                                     anomal_training=True,
                                     latent_res=64,
                                     do_anomal_sample=True)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    check_base_dir = f'/home/dreamyou070/dataset_check_folder/{obj_name}/anomal_p_{args.anomal_p}'
    os.makedirs(check_base_dir, exist_ok=True)

    for sample in train_dataloader :
        name = sample['image_name'][0]
        save_name = sample['anomal_name'][0]


        image = sample['image'].squeeze() # [3,512,512]
        np_img = np.array(((image + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_image = Image.fromarray(np_img)
        pil_image.save(os.path.join(check_base_dir, f'{save_name}.png'))
        object_mask = sample['object_mask']
        np_object_mask = object_mask.squeeze().numpy()
        pil_object_mask = Image.fromarray((np_object_mask * 255).astype(np.uint8))
        pil_object_mask.save(os.path.join(check_base_dir, f'{save_name}_object_mask.png'))

        merged_src = sample['anomal_image'].squeeze()
        np_merged_src = np.array(((merged_src + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_merged_src = Image.fromarray(np_merged_src)
        pil_merged_src.save(os.path.join(check_base_dir, f'{save_name}_anomal_aug.png'))
        anomaly_mask = sample['anomal_mask']
        np_anomaly_mask = anomaly_mask.squeeze().numpy()
        pil_anomaly_mask = (np_anomaly_mask * 255).astype(np.uint8)
        pil_anomaly_mask = Image.fromarray(pil_anomaly_mask)
        pil_anomaly_mask.save(os.path.join(check_base_dir, f'{save_name}_anomal_aug_mask.png'))

        merged_src = sample['bg_anomal_image'].squeeze()
        np_merged_src = np.array(((merged_src + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_merged_src = Image.fromarray(np_merged_src)
        pil_merged_src.save(os.path.join(check_base_dir, f'{save_name}_bg.png'))
        anomaly_mask = sample['bg_anomal_mask']
        np_anomaly_mask = anomaly_mask.squeeze().numpy()
        pil_anomaly_mask = (np_anomaly_mask * 255).astype(np.uint8)
        pil_anomaly_mask = Image.fromarray(pil_anomaly_mask)
        pil_anomaly_mask.save(os.path.join(check_base_dir, f'{save_name}_bg_mask.png'))

        """
        merged_src = sample['rotate_image'].squeeze()
        np_merged_src = np.array(((merged_src + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_merged_src = Image.fromarray(np_merged_src)
        pil_merged_src.save(os.path.join(check_base_dir, f'{save_name}_rotate.png'))

        anomaly_mask = sample['rotate_mask']
        np_anomaly_mask = anomaly_mask.squeeze().numpy()
        pil_anomaly_mask = (np_anomaly_mask * 255).astype(np.uint8)
        pil_anomaly_mask = Image.fromarray(pil_anomaly_mask)
        pil_anomaly_mask.save(os.path.join(check_base_dir, f'{save_name}_rotate_mask.png'))
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument('--obj_name', type=str, default='transistor')
    parser.add_argument('--anomaly_source_path', type=str,default=r'/home/dreamyou070/MyData/anomal_source')
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--anomal_p", type=float, default=0.03)

    parser.add_argument("--anomal_min_perlin_scale", type=int, default=0)
    parser.add_argument("--anomal_max_perlin_scale", type=int, default=3)
    parser.add_argument("--anomal_min_beta_scale", type=float, default=0.5)
    parser.add_argument("--anomal_max_beta_scale", type=float, default=0.8)


    parser.add_argument("--back_min_perlin_scale", type=int, default=0)
    parser.add_argument("--back_max_perlin_scale", type=int, default=3)
    parser.add_argument("--back_min_beta_scale", type=float, default=0.6)
    parser.add_argument("--back_max_beta_scale", type=float, default=0.9)
    parser.add_argument("--do_rot_augment", action='store_true')
    parser.add_argument("--anomal_trg_beta", type=float)
    parser.add_argument("--back_trg_beta", type=float)

    args = parser.parse_args()
    passing_mvtec_argument(args)
    main(args)