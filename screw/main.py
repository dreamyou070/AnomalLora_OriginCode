import os
from PIL import Image
import numpy as np

base_folder = '/home/dreamyou070/MyData/anomaly_detection/MVTec/screw'
ground_truth_folder = os.path.join(base_folder, 'ground_truth')
test_folder = os.path.join(base_folder, 'test')
defects = os.listdir(ground_truth_folder)
for defect in defects:
    defect_folder = os.path.join(ground_truth_folder, defect)
    test_defect_folder = os.path.join(test_folder, defect)
    test_gt_folder = os.path.join(test_defect_folder, 'gt')
    os.makedirs(test_gt_folder, exist_ok=True)
    images = os.listdir(defect_folder)
    for image in images:
        new_name = image.replace('_mask', '')
        org_path = os.path.join(defect_folder, image)
        new_path = os.path.join(test_gt_folder, new_name)
        img = Image.open(org_path).convert('L')
        img.save(new_path)