from removebg import RemoveBg
import os
base_dir = r'/home/dreamyou070/MyData/anomaly_detection/MVTec/transistor/train/good'

# [1]
rgb_folder = os.path.join(base_dir, 'rgb')
# [2]
object_folder = os.path.join(base_dir, 'background_removed_rgb')
os.makedirs(object_folder, exist_ok=True)

# [3]
images = os.listdir(rgb_folder)
api_key = 'iuUxT3T8g4BeqasbkP5EPghZ'

rmbg = RemoveBg(api_key, "error.log")
for image in images:
    img_dir = os.path.join(rgb_folder, image)
    rmbg.remove_background_from_img_file(img_dir)
    rmbg.save(os.path.join(object_folder, image))
    rmbg.close()
    print(f'processed {image}')
    break