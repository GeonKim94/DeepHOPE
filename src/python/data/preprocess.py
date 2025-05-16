import random

import torch

import numpy as np

from scipy import ndimage

def clip(img, ri=None):
    if img.dtype == 'uint8': #for brightfield
        cap_min = 0
        cap_max = 191
    else:
        cap_min = 13300 if np.max(img) > 10000 else 1.3300
        cap_max = 14000 if np.max(img) > 10000 else 1.4000
    img = np.clip((img.astype('float' - cap_min) / (cap_max - cap_min), 0, 1)
    return img


def bottom_crop_z(img, z_shape):
    padval = 13370.0 if np.max(img) > 10000 else 1.3370
    z_range = [i for i in range(z_shape)]
    if z_shape > img.shape[2]:
        pad_width = ((0, 0),
             (0, 0), 
             (0, z_shape-img.shape[2])) 
        return np.pad(img, pad_width=pad_width, mode='constant', constant_values=padval)
    else:
        return img[:,:,z_range]

def random_crop_xy(img, target_shape):
    padval = 13370.0 if np.max(img) > 10000 else 1.3370

    origin_size = img.shape

    # Crop or pad along the first dimension (height)
    if origin_size[0] > target_shape[0]:
        rand_x = random.randint(0, origin_size[0] - target_shape[0])
        img = img[rand_x:rand_x + target_shape[0], :, :]
    else:
        pad_len = target_shape[0] - origin_size[0]
        pad_len_front = random.randint(0, pad_len)
        pad_len_back = pad_len - pad_len_front
        img = np.pad(img, ((pad_len_front, pad_len_back), (0, 0), (0, 0)), constant_values=padval)

    # Crop or pad along the second dimension (width)
    if origin_size[1] > target_shape[1]:
        rand_y = random.randint(0, origin_size[1] - target_shape[1])
        img = img[:, rand_y:rand_y + target_shape[1], :]
    else:
        pad_len = target_shape[1] - origin_size[1]
        pad_len_front = random.randint(0, pad_len)
        pad_len_back = pad_len - pad_len_front
        img = np.pad(img, ((0, 0), (pad_len_front, pad_len_back), (0, 0)), constant_values=padval)

    return img

def center_crop_xy(img, target_shape):
    origin_size = img.shape
    padval = 1.3370 if np.max(img) <= 10000 else 13370.
    
    if origin_size[0] > target_shape[0]:
        rand_x = (origin_size[0] // 2) - (target_shape[0] // 2)
        img = img[rand_x:rand_x + target_shape[0], :, :]
    else:
        pad_len = target_shape[0] - origin_size[0]
        pad_len_front = pad_len // 2
        pad_len_back = pad_len - pad_len_front
        img = np.pad(img, ((pad_len_front, pad_len_back), (0, 0), (0, 0)), constant_values=padval)
        
    if origin_size[1] > target_shape[1]:
        rand_y = (origin_size[1] // 2) - (target_shape[1] // 2)
        img = img[:, rand_y:rand_y + target_shape[1], :]
    else:
        pad_len = target_shape[1] - origin_size[1]
        pad_len_front = pad_len // 2
        pad_len_back = pad_len - pad_len_front
        img = np.pad(img, ((0, 0), (pad_len_front, pad_len_back), (0, 0)), constant_values=padval)

    return img


def z_to_ch(input):
    input = np.swapaxes(input,0,2)
    return input

def single_ch(input):
    input = np.expand_dims(input, 0)
    return input


def gaussian_noise(img):
    sigma = random.uniform(0.001, 0.01)
    noise = np.random.normal(0, sigma, size=img.shape)

    if isinstance(img, list):
        return [i + noise for i in img]

    return img + noise

def flip_x(img):
    if np.random.randint(0, 2) == 0:
        img = img[::-1, :, :].copy()
    return img

def flip_y(img):
    if np.random.randint(0, 2) == 0:
        img = img[:, ::-1, :].copy()
    return img

def rotate_xy(img):
    angle = random.randrange(0, 360, 90)
    return ndimage.interpolation.rotate(img, angle,
                                        reshape=False,
                                        order=0,
                                        mode='reflect')

def swapaxes_xy(img):
    if np.random.randint(0, 2) == 0:
        img = np.swapaxes(img, 0, 1)
    return img

def to_tensor(img):
    out = torch.from_numpy(img).float()
    return out
