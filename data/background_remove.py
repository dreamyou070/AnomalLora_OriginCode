from PIL import Image
from rembg import remove
import numpy as np

input_path = '003.png'
output_path = 'output.png'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)

output_pil = Image.open(output_path)
output_np = np.array(output_pil)
alpha_channel = output_np[:, :, 3]
alpha_channel = np.where(alpha_channel == 0, 0,1)
mask = Image.fromarray(alpha_channel * 255)
mask.show()