import os
from PIL import Image
import numpy as np
png_folder = 'png_object'
images = os.listdir(png_folder)
for image in images:
    img_dir = os.path.join(png_folder, image)
    pil_img = Image.open(img_dir)
    alpha = np.array(pil_img)[:,:,-1]
    alpha_img = Image.fromarray(alpha * 255)
    alpha_img.show()