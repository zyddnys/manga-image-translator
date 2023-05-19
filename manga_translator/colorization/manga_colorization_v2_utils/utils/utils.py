import numpy as np
import cv2

def resize_pad(img, size = 256):
            
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        
    if img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
        
    if img.shape[2] == 4:
        img = img[:, :, :3]

    pad = None        
            
    if (img.shape[0] < img.shape[1]):
        height = img.shape[0]
        ratio = height / (size * 1.5)
        width = int(np.ceil(img.shape[1] / ratio))
        img = cv2.resize(img, (width, int(size * 1.5)), interpolation = cv2.INTER_AREA)

        
        new_width = width + (32 - width % 32)
            
        pad = (0, new_width - width)
        
        img = np.pad(img, ((0, 0), (0, pad[1]), (0, 0)), 'maximum')
    else:
        width = img.shape[1]
        ratio = width / size
        height = int(np.ceil(img.shape[0] / ratio))
        img = cv2.resize(img, (size, height), interpolation = cv2.INTER_AREA)

        new_height = height + (32 - height % 32)
            
        pad = (new_height - height, 0)
        
        img = np.pad(img, ((0, pad[0]), (0, 0), (0, 0)), 'maximum')
        
    if (img.dtype == 'float32'):
        np.clip(img, 0, 1, out = img)

    return img[:, :, :1], pad
