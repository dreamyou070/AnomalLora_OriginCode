import torch
import numpy as np
from PIL import Image
import cv2

end_num = 512
x = np.arange(0, end_num, 1, float)
y = np.arange(0, end_num, 1, float)[:, np.newaxis]
x_center = torch.randint(int(end_num / 4), int(3 * end_num / 4),(1,)).item()
y_center = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
#sigma = torch.randint(25, 160, (1,)).item()
sigma = 100
result = np.exp(-4 * np.log(2) * ((x - x_center) ** 2 + (y - y_center) ** 2) / sigma ** 2)
result_pil = Image.fromarray((result * 255).astype(np.uint8)).convert('L')
result_pil.show()

"""
def gaussian_augment_image(self, image, back_img, object_position):
    while True:
        while True:
            end_num = self.resize_shape[0]  # 512
            x = np.arange(0, end_num, 1, float)
            y = np.arange(0, end_num, 1, float)[:, np.newaxis]
            x_0 = torch.randint(int(end_num / 4),
                                int(3 * end_num / 4),
                                (1,)).item()
            
            y_0 = torch.randint(int(end_num / 4), int(3 * end_num / 4), (1,)).item()
            
            sigma = torch.randint(25, 60, (1,)).item()
            
            result = np.exp(-4 * np.log(2) * (     (x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2       )  # 0 ~ 1
            
            result_thr = np.where(result < 0.5, 0, 1).astype(np.float32)
            result_thr = cv2.GaussianBlur(result_thr, (5, 5), 0) # hole part
            if object_position is not None:
                total_object_pixel = np.sum(object_position)
                blur_2D_mask = (result_thr * object_position).astype(np.float32)
            binary_2D_mask = (np.where(blur_2D_mask == 0, 0, 1)).astype(np.float32)  # [512,512,3] # where to hole part
            if np.sum(binary_2D_mask) > anomal_p * total_object_pixel:
                break
        blur_3D_mask = np.expand_dims(blur_2D_mask, axis=2)  # [512,512,3]
        A = back_img.astype(np.float32)  # merge with zero back img
        augmented_image = (image * (1 - blur_3D_mask) + A * blur_3D_mask).astype(np.float32) # original img + hole

        anomal_img = np.array(Image.fromarray(augmented_image.astype(np.uint8)), np.uint8)

        binary_2d_pil = Image.fromarray((binary_2D_mask * 255).astype(np.uint8)).convert('L').resize((64, 64))
        anomal_mask_torch = torch.where((torch.tensor(np.array(binary_2d_pil)) / 255) > 0.5, 1, 0)
        if anomal_mask_torch.sum() > 0:
            break
    return anomal_img, anomal_mask_torch
"""