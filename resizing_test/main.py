import os
from PIL import Image
import numpy as np

base_folder = 'folder'
rgb_dir = os.path.join(base_folder, f'000_rgb.png')
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
cropped_img = pil_img.crop((w_start, h_start, w_end, h_end)).resize((512,512))

pil_rgb = Image.open(rgb_dir).crop((w_start, h_start, w_end, h_end)).resize((512,512))
pil_rgb.show()

