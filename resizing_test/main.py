import os
from PIL import Image
import numpy as np

base_folder = 'folder'

# [1]
rgb_dir = os.path.join(base_folder, 'rgb_021.png')
rgb_pil = Image.open(rgb_dir)

# [2]
anomal_pil = Image.rotate(rgb_pil, 45)
anomal_pil.show()
