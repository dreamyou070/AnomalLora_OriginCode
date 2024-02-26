import numpy as np
import cv2
import torch
from PIL import Image
min_sigma = 90
max_sigma = 100
end_num = 512
x = np.arange(0, end_num, 1, float)
y = np.arange(0, end_num, 1, float)[:, np.newaxis]
x_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
y_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
# if sigmal big -> big circle
# if sigmal small -> small circle
sigma = torch.randint(min_sigma, max_sigma, (1,)).item()

result = np.exp(-4 * np.log(2) * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)  # 0 ~ 1
result_thr = np.where(result < 0.5, 0, 1).astype(np.float32)
result_thr = cv2.GaussianBlur(result_thr, (5,5), 0)
Image.fromarray((result_thr * 0.6 * 255).astype(np.uint8)).show()