import os
from PIL import Image
import numpy as np

mask_base_folder = r'jpg_object_mask'
mask_1 = os.path.join(mask_base_folder, 'mask_1')
mask_2 = os.path.join(mask_base_folder, 'mask_2')
mask_3 = os.path.join(mask_base_folder, 'mask_3')

imgs = os.listdir(mask_1)
for img in imgs:
    img_1_dir = os.path.join(mask_1, img)
    img_2_dir = os.path.join(mask_2, img)
    img_3_dir = os.path.join(mask_3, img)

    img_1_np = 255 - np.array(Image.open(img_1_dir).convert('L'))
    img_2_np = 255 - np.array(Image.open(img_2_dir).convert('L'))
    img_3_np = 255 - np.array(Image.open(img_3_dir).convert('L'))
    final_np = img_1_np + img_2_np + img_3_np
    final_np = np.where(final_np == 0 , 0, 255)
    final_pil = Image.fromarray(final_np).convert('L')
    final_pil.show()
