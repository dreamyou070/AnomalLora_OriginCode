import os
import numpy as np
from PIL import Image
from rembg import remove

base_folder = 'train'

rgb_org_folder = os.path.join(base_folder, 'rgb_origin')
mask_folder = os.path.join(base_folder, 'bg_corrected_rgb')

rgb_folder = os.path.join(base_folder, 'rgb')

imgs = os.listdir(rgb_org_folder)
for img in imgs:
    rgb_path = os.path.join(rgb_org_folder, img)
    rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
    black_mask = rgb_np * 0

    mask_path = os.path.join(mask_folder, img)
    mask_pil = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask_pil) / 255

    new_np = rgb_np * mask_np + black_mask * (1 - mask_np)
    new_pil = Image.fromarray(new_np.astype('uint8'), 'RGB')
    new_pil.save(os.path.join(rgb_folder, img))