import os
from PIL import Image
import numpy as np

base_folder = 'folder'

# [1]
rgb_dir = os.path.join(base_folder, f'000_rgb.png')
pil_img = Image.open(rgb_dir).convert('RGB')
org_h, org_w = pil_img.size

# [2]
mask_dir = os.path.join(base_folder, f'000_mask.png')

pil_img = Image.open(mask_dir)
np_img = np.array(pil_img)
h,w = np_img.shape
h_indexs, w_indexs = [], []
for h_i in range(h):
    for w_i in range(w):
        if np_img[h_i, w_i] > 0:
            h_indexs.append(h_i)
            w_indexs.append(w_i)
h_start, h_end = min(h_indexs), max(h_indexs)
w_start, w_end = min(w_indexs), max(w_indexs)
trg_h, trg_w = h_end - h_start, w_end - w_start
input_img = pil_img.crop((w_start, h_start, w_end, h_end))#.resize((512,512))
trg_h, trg_w = input_img.size

# [3]
basic_cls_map_pil = Image.new('L', (org_h, org_w), 0)
basic_normal_map_pil = Image.new('L', (org_h, org_w), 255)
basic_anomaly_map_pil = Image.new('L', (org_h, org_w), 0)
basic_cls_map_pil.paste(input_img, (w_start, h_start, w_end, h_end))
basic_normal_map_pil.paste(input_img, (w_start, h_start, w_end, h_end))
basic_anomaly_map_pil.paste(input_img, (w_start, h_start, w_end, h_end))
basic_normal_map_pil.show()
