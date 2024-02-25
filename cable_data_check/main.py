import os
import numpy as np
from PIL import Image
from rembg import remove
def remove_background(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

base_folder = 'train'

rgb_org_folder = os.path.join(base_folder, 'rgb_origin')
object_mask_folder = os.path.join(base_folder, 'object_mask')
bg_corrected_folder = os.path.join(base_folder, 'bg_corrected')
os.makedirs(bg_corrected_folder, exist_ok=True)
bg_corrected_rgb_folder = os.path.join(base_folder, 'bg_corrected_rgb')
os.makedirs(bg_corrected_rgb_folder, exist_ok=True)

#first_back_remove_folder = os.path.join(base_folder, 'rgb_remove_background_1')
#os.makedirs(first_back_remove_folder, exist_ok=True)
#second_back_remove_folder = os.path.join(base_folder, 'rgb_remove_background')
#os.makedirs(second_back_remove_folder, exist_ok=True)

#imgs = os.listdir(rgb_org_folder)
imgs = os.listdir(object_mask_folder)
for img in imgs:

    org_mask_dir = os.path.join(object_mask_folder, img)
    new_mask_dir = os.path.join(bg_corrected_folder, img)
    remove_background(org_mask_dir, new_mask_dir)
    Image.open(new_mask_dir).convert("RGB").save(os.path.join(bg_corrected_rgb_folder, img))