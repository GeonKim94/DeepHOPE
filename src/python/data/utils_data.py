import numpy as np
import torch
from einops import rearrange
from os.path import basename, splitext, isfile, isdir
from PIL import Image
import h5py
from scipy.io import loadmat
from scipy.ndimage.interpolation import rotate
import random


def readimg(path_):
    ext_path_ = splitext(path_)[-1]
    if ext_path_ in ['.png', '.jpg', '.tif']:
        data_ = Image.open(path_).convert("RGB")
        data_ = np.array(data_)
        #data_ = rearrange(data_, "h w c -> c h w")
    elif ext_path_ == '.mat':
        dict_data = loadmat(path_)
        data_ = dict_data['ri']
    elif ext_path_ == '.h5':
        file_data = h5py.File(path_,'r')
        data_ = np.array(file_data.get('/ri'))
    else:
        ValueError(f'Image file extension "{ext_path_}" is not supported')
    return np.array(data_).astype('float16')

def one_hot(N, y):
    if not (0 <= y < N):
        raise ValueError(f"Index y={y} is out of bounds for size N={N}.")
    one_hot_vector = torch.zeros((1, N), dtype=torch.float32)  # Shape (1, N)
    one_hot_vector[0, y] = 1.0  # Set the correct index to 1
    return one_hot_vector

def pad(img, target_shape):
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Invalid image dimensions: either height or width is zero.")

    origin_size = img.shape
    pad_xf = pad_xb = pad_yf = pad_yb = 0

    if target_shape[0] > origin_size[0]:
        pad_xf = (target_shape[0] - origin_size[0]) // 2
        pad_xb = target_shape[0] - origin_size[0] - pad_xf

    if target_shape[1] > origin_size[1]:
        pad_yf = (target_shape[1] - origin_size[1]) // 2
        pad_yb = target_shape[1] - origin_size[1] - pad_yf

    pad_zf, pad_zb = 0, 0
    pad_width = ((pad_xf, pad_xb), (pad_yf, pad_yb), (pad_zf, pad_zb))

    pad_val = 0
    
    img = np.pad(img, pad_width, mode="constant", constant_values=pad_val)
    return img

def random_crop(img, target_shape):

    img = pad(img,target_shape)
    origin_size = img.shape

    rand_x = random.randint(0, origin_size[0] - target_shape[0])
    img = img[rand_x:rand_x + target_shape[0], :, :]

    rand_y = random.randint(0, origin_size[1] - target_shape[1])
    img = img[:, rand_y:rand_y + target_shape[1], :]
  
    return img

def center_crop(img, target_shape):
    img = pad(img,target_shape)
    origin_size = img.shape
    
    rand_x = (origin_size[0] // 2) - (target_shape[0] // 2)
    img = img[rand_x:rand_x + target_shape[0], :, :]
    
    rand_y = (origin_size[1] // 2) - (target_shape[1] // 2)
    img = img[:, rand_y:rand_y + target_shape[1], :]

    return img

def random_gaussian(img):
    sigma = random.uniform(0, 5.)
    noise = np.random.normal(0, sigma, size=img.shape)
    return img + noise

def random_flip(img):
    rand = random.randint(0, 1)
    if rand == 0:
        img = img[::-1, :].copy()
    rand = random.randint(0, 1)
    if rand == 0:
        return img[:, ::-1].copy()
    else:
        return img

def random_rotate(img):
    angle = random.randrange(0, 360, 90)
    return rotate(img, angle,
                    reshape=False,
                    order=0,
                    mode='reflect')

def random_swapaxes(img):
    rand = random.randint(0, 1)
    if rand == 0:
        img = np.swapaxes(img, 0, 1)
    return img

def to_tensor(img):
    img = np.swapaxes(img,0,2)
    return torch.from_numpy(img).float()