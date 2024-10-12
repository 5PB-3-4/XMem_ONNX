import numpy as np
import cv2

def compute_tensor_iu(seg: np.ndarray, gt: np.ndarray):
    intersection = (seg & gt).astype(np.float32).sum()
    union = (seg | gt).astype(np.float32).sum()

    return intersection, union

def compute_tensor_iou(seg: np.ndarray, gt: np.ndarray):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou 

# STM
def pad_divide_by(in_img: np.ndarray, d: int):
    h, w = in_img.shape[:2]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))

    pad_value = (0, 0, 0) if len(in_img.shape)==3 else 0
    out = cv2.copyMakeBorder(in_img, top=lh,bottom=uh,left=lw,right=uw,borderType=cv2.BORDER_CONSTANT, value=pad_value)
    return out, pad_array

def unpad(img, pad):
    if len(img.shape) == 4:
        if pad[2]+pad[3] > 0:
            img = img[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,:,pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2]+pad[3] > 0:
            img = img[:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,pad[0]:-pad[1]]
    else:
        raise NotImplementedError
    return img