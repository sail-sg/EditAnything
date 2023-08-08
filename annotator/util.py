import numpy as np
import cv2
import os
import pickle

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def resize_points(clicked_points, original_shape, resolution):
    original_height, original_width, _ = original_shape
    original_height = float(original_height)
    original_width = float(original_width)
    
    scale_factor = float(resolution) / min(original_height, original_width)
    resized_points = []
    
    for point in clicked_points:
        x, y, lab = point
        resized_x = int(round(x * scale_factor))
        resized_y = int(round(y * scale_factor))
        resized_point = (resized_x, resized_y, lab)
        resized_points.append(resized_point)
    
    return resized_points

def get_bounding_box(mask):
    # Convert PIL Image to numpy array
    mask = np.array(mask).astype(np.uint8)

    # Take the first channel (R) of the mask
    mask = mask[:,:,0]

    # Get the indices of elements that are not zero
    rows = np.any(mask, axis=0)
    cols = np.any(mask, axis=1)
    
    # Get the minimum and maximum indices where the elements are not zero
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Return as [xmin, ymin, xmax, ymax]
    return [rmin, cmin, rmax, cmax]



def save_input_to_file(func):
    def wrapper(self, *args, **kwargs):
        # 创建不包含 self 的输入副本
        input_data = {
            'args': args,
            'kwargs': kwargs
        }
        
        # 执行原始函数
        result = func(self, *args, **kwargs)
        
        # 将输入数据保存到文件
        with open('input_data.pkl', 'wb') as f:
            pickle.dump(input_data, f)
        
        # 返回结果
        return result

    return wrapper
