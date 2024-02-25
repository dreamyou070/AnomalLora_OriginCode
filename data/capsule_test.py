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

                # [1] new files
                rgb_dir = os.path.join(defect_dir, 'rgb_origin')
                origin_folder = os.path.join(defect_dir, 'rgb_origin')
                os.makedirs(origin_folder, exist_ok=True)
                sub_folder = os.path.join(defect_dir, 'rgb_remove_background')
                os.makedirs(sub_folder, exist_ok=True)
                gt_folder = os.path.join(defect_dir, 'gt')

                imgs = os.listdir(rgb_dir)
                for img in imgs:
                    # [1] save original image
                    origin_rgb_dir = os.path.join(rgb_dir, img)
                    Image.open(origin_rgb_dir).save(os.path.join(origin_folder, img))

                    # [2] remove background
                    sub_dir = os.path.join(sub_folder, img)
                    remove_background(origin_rgb_dir, sub_dir)
                    Image.open(sub_dir).convert("RGB").save(os.path.join(rgb_dir, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str,
                        default=r'/home/dreamyou070/MyData/anomaly_detection/MVTec')
    parser.add_argument('--trg_cat', type=str, default='capsule')
    args = parser.parse_args()
    main(args)