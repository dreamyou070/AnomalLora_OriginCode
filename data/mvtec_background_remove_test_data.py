from PIL import Image
from rembg import remove
import numpy as np
import os
import argparse


def remove_background(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)


def main(args):
    print(f'step 2. prepare images')
    base_folder = args.base_folder
    cats = os.listdir(base_folder)

    for cat in cats:
        if cat == args.trg_cat:

            cat_dir = os.path.join(base_folder, f'{cat}')
            test_dir = os.path.join(cat_dir, 'test')
            ground_truth_dir = os.path.join(cat_dir, 'ground_truth')

            defetcs = os.listdir(test_dir)

            for defect in defetcs:

                defect_dir = os.path.join(test_dir, f'{defect}')
                gt_defect_dir = os.path.join(ground_truth_dir, f'{defect}')

                imgs = os.listdir(gt_defect_dir)

                for img in imgs:

                    # [1] new folder
                    rgb_dir = os.path.join(defect_dir, 'rgb')
                    os.makedirs(rgb_dir, exist_ok=True)
                    origin_folder = os.path.join(defect_dir, 'rgb_origin')
                    os.makedirs(origin_folder, exist_ok=True)
                    sub_folder = os.path.join(defect_dir, 'rgb_remove_background')
                    os.makedirs(sub_folder, exist_ok=True)
                    gt_folder = os.path.join(defect_dir, 'gt')
                    os.makedirs(gt_folder, exist_ok=True)

                    # [1] save original image
                    img_dir = os.path.join(gt_defect_dir, img)
                    pil_img = Image.open(img_dir)
                    origin_img_dir = os.path.join(origin_folder, image)
                    pil_img.save(origin_img_dir)

                    # [2] remove background
                    sub_dir = os.path.join(sub_folder, image)
                    remove_background(img_dir, sub_dir)
                    background_removed_img = Image.open(sub_dir).convert("RGB")
                    background_removed_img.save(img_dir)

                    # [3] copy to rgb folder
                    original_gt_dir = os.path.join(gt_defect_dir, image)
                    new_gt_dir = os.path.join(gt_folderm, image)
                    Image.open(original_gt_dir).convert("L").save(new_gt_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='bottle')
    args = parser.parse_args()
    main(args)