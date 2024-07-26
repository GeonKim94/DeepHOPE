import random

import torch

import numpy as np

from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from .preprocess3d import calibration

#import matplotlib.pyplot as plt


#crop_shape = (256, 256)  # before 230307 (original img 320)
#size_z = 32 # before 230307
crop_unit = (64, 64)  # -> variable input size doesn't work for batch approach
crop_min = (512, 512)  # -> variable input size doesn't work for batch approach

#crop_shape = (3072,3072)
crop_shape = (2048,2048)
size_z = 32


def random_crop_2d(img, target_shape=crop_shape, **kwargs):
    if np.max(img) > 10000:
        padval = 13370.
    else:
        padval = 1.3370
    origin_size = img.shape
    
    if origin_size[0] > target_shape[0]:
        rand_x = random.randint(0, origin_size[0] - target_shape[0])
        img = img[rand_x:rand_x + target_shape[0],:]
    else:
        pad_len = target_shape[0] - origin_size[0] 
        pad_len_front = random.randint(0, pad_len) 
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((padval*np.ones((pad_len_front,img.shape[1])),
            img, padval*np.ones((pad_len_back,img.shape[1]))), axis = 0)

    if origin_size[1] > target_shape[1]:
        rand_y = random.randint(0, origin_size[1] - target_shape[1])
        img = img[:,rand_y:rand_y + target_shape[1]]
    else:
        pad_len = target_shape[1] - origin_size[1] 
        pad_len_front = random.randint(0, pad_len) 
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((padval*np.ones((img.shape[0],pad_len_front)),
            img, padval*np.ones((img.shape[0],pad_len_back))), axis = 1)
        
    return img


def center_crop_2d(img, target_shape=crop_shape, **kwargs):
    if np.max(img) > 10000:
        padval = 13370.
    else:
        padval = 1.3370
    origin_size = img.shape
    
    if origin_size[0] > target_shape[0]:
        rand_x = origin_size[0]//2-target_shape[0]//2
        img = img[rand_x:rand_x + target_shape[0],:]
    else:
        pad_len = target_shape[0] - origin_size[0] 
        pad_len_front = pad_len//2
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((padval*np.ones((pad_len_front,img.shape[1])),
            img, padval*np.ones((pad_len_back,img.shape[1]))), axis = 0)

    if origin_size[1] > target_shape[1]:
        rand_y = origin_size[1]//2-target_shape[1]//2
        img = img[:,rand_y:rand_y + target_shape[1]]
    else:
        pad_len = target_shape[1] - origin_size[1] 
        pad_len_front = pad_len//2
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((padval*np.ones((img.shape[0],pad_len_front)),
            img, padval*np.ones((img.shape[0],pad_len_back))), axis = 1)
        
    return img


## this could not be used because of the batch approach
#def random_crop_2d(img, crop_unit = crop_unit, crop_min = crop_min, **kwargs):
#    origin_size = img.shape
#    rand_x = random.randint(0 + crop_min[0]//2, origin_size[0] - crop_min[0]//2)
#    rand_y = random.randint(0 + crop_min[1]//2, origin_size[1] - crop_min[1]//2)
#    
#    x_size_crop = 2*min(rand_x, origin_size[0]-rand_x)//64 * 64
#    y_size_crop = 2*min(rand_y, origin_size[1]-rand_y)//64 * 64
#
#    img = img[rand_x-x_size_crop//2 : rand_x+x_size_crop//2, rand_y-y_size_crop//2 : rand_y+y_size_crop//2]  # 75 +- size_z
#    return img
#
#def center_crop_2d(img, crop_unit = crop_unit, crop_min = crop_min, **kwargs):
#    origin_size = img.shape
#    rand_x = origin_size[0]//2
#    rand_y = origin_size[1]//2    
#    
#    x_size_crop = 2*min(rand_x, origin_size[0]-rand_x)//64 * 64
#    y_size_crop = 2*min(rand_y, origin_size[1]-rand_y)//64 * 64
#
#    img = img[rand_x-x_size_crop//2 : rand_x+x_size_crop//2, rand_y-y_size_crop//2 : rand_y+y_size_crop//2]  # 75 +- size_z
#    return img


def gaussian_2d(img, **kwargs):
    sigma = random.uniform(0.001, 0.01)
    noise = np.random.normal(0, sigma, size=img.shape)

    if isinstance(img, list):
        return [i + noise for i in img]

    return img + noise


def flipud_2d(img, **kwargs):
    rand = random.randint(0, 1)
    if isinstance(img, list):
        if rand == 0:
            return [i[::-1, :].copy() for i in img]
        else:
            return img

    if rand == 0:
        return img[::-1, :].copy()
    else:
        return img


def fliplr_2d(img, **kwargs):
    rand = random.randint(0, 1)
    if isinstance(img, list):
        if rand == 0:
            return [i[:, ::-1].copy() for i in img]
        else:
            return img

    if rand == 0:
        return img[:, ::-1].copy()
    else:
        return img


# https://github.com/scipy/scipy/issues/5925
def rotate_2d(img, **kwargs):
    angle = random.randrange(0, 360, 90)
    if isinstance(img, list):
        return [ndimage.interpolation.rotate(i, angle,
                                             reshape=False,
                                             order=0,
                                             mode='reflect') for i in img]

    # rand = random.randint(1,360)
    return ndimage.interpolation.rotate(img, angle,
                                        reshape=False,
                                        order=0,
                                        mode='reflect')


def to_tensor_2d(img, **kwargs):
    if isinstance(img, list):
        img = np.stack(img, axis=0)

    out = torch.from_numpy(img).float()
    if len(out.shape) == 2:
        out = out.unsqueeze(0)
    return out


# (1, 1), (5, 2), (1, 0.5), (1, 3)
def elastic_transform_2d(img, alpha=0, sigma=0, random_state=None, **kwargs):

    param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]
    if alpha == 0 and sigma == 0:
        rand = random.randint(0, 3)
        alpha, sigma = param_list[rand]

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    if isinstance(img, list):
        new_imgs = []
        for i in img:
            new_imgs.append(elastic_transform(i, alpha=alpha, sigma=sigma))
        return new_imgs

    trasform_img = map_coordinates(img, indices, order=1).reshape(shape)
    #transform_img = np.zeros(img.shape)
    #for i in range(img.shape[2]):
    #    trasform_img[:, :, i] = map_coordinates(img[:, :, i], indices, order=1).reshape(shape)

    return trasform_img

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, ri=None):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W, Z).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        H = img.size(1)
        W = img.size(2)
        Z = img.size(3)

        mask = np.ones((H, W, Z), np.float32)

        for n in range(self.n_holes):
            z = np.random.randint(Z)
            y = np.random.randint(H)
            x = np.random.randint(W)

            y1 = np.clip(y - self.length // 2, 0, H)
            y2 = np.clip(y + self.length // 2, 0, H)
            x1 = np.clip(x - self.length // 2, 0, W)
            x2 = np.clip(x + self.length // 2, 0, W)
            z1 = np.clip(z - self.length // 2, 0, Z)
            z2 = np.clip(z + self.length // 2, 0, Z)

            mask[y1: y2, x1: x2, z1:z2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


TRAIN_AUGS_2D = [
    random_crop_2d,
    # center_crop_3d,
    # multi_crop_3d,
    calibration,
    gaussian_2d,
    #elastic_transform_2d,
    flipud_2d,
    fliplr_2d,
    rotate_2d,
    to_tensor_2d,
    # Cutout(3, 8)
]

TEST_AUGS_2D = [
    center_crop_2d,
    calibration,
    # center_multi_crop_3d,
    to_tensor_2d
]

