import os
from PIL import Image
import numpy as np
import cv2

trg_img_dir = '002.png'
image = cv2.imread(trg_img_dir)
blank_mask = np.zeros(image.shape, dtype=np.uint8)
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([18, 42, 69])
upper = np.array([179, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    cv2.drawContours(blank_mask,[c], -1, (255,255,255), -1)
    break

result = cv2.bitwise_and(original,blank_mask)

cv2.imshow('result', result)
cv2.waitKey()