import numpy as np
import cv2

im_mean = (124, 116, 104)

def im_normalization(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = (image / 255.0 - mean) / std
    input_img = input_img.astype(np.float32)
    return input_img

def inv_im_trans(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean = np.array([-0.485/0.229, -0.456/0.224, -0.406/0.225])
    std = np.array([1/0.229, 1/0.224, 1/0.225])
    input_img = (image / 255.0 - mean) / std
    input_img = input_img.astype(np.float32)
    return input_img