import random

import torch

import numpy as np

from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#import matplotlib.pyplot as plt


#crop_shape = (256, 256)  # before 230307 (original img 320)
#size_z = 32 # before 230307
#crop_shape = (384, 384)  # since 230307 (original img 512)
crop_shape = (2048, 2048) # since 230502
size_z = 8

def mat2npy(mat, **kwargs):
    #img = mat['dataRaw'] # good data
    img = mat['data'] # ys kim data (1:1:1:1:1:1 split data)
    #img = mat['RItomogram']
    # img = np.float16(mat['data'])
    #ri = mat['n_m'] # good data 
    ri = 1.3374#mat['ri'] # ys kim data (evenly mixed data)
    #ri = 1.33
    return img, ri


def calibration(img, ri):
    img = (img - ri) * 9
    img[img < 0] = 0
    return img



def random_crop_3d(img, target_shape=crop_shape, **kwargs):
    origin_size = img.shape
    
    if origin_size[0] > target_shape[0]:
        rand_x = random.randint(0, origin_size[0] - target_shape[0])
        img = img[rand_x:rand_x + target_shape[0],:,:]
    else:
        pad_len = target_shape[0] - origin_size[0] 
        pad_len_front = random.randint(0, pad_len) 
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((1.337*np.ones((pad_len_front,img.shape[1],img.shape[2])),
            img, 1.337*np.ones((pad_len_back,img.shape[1],img.shape[2]))), axis = 0)

    if origin_size[1] > target_shape[1]:
        rand_y = random.randint(0, origin_size[1] - target_shape[1])
        img = img[:,rand_y:rand_y + target_shape[1],:]
    else:
        pad_len = target_shape[1] - origin_size[1] 
        pad_len_front = random.randint(0, pad_len) 
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((1.337*np.ones((img.shape[0],pad_len_front,img.shape[2])),
            img, 1.337*np.ones((img.shape[0],pad_len_back,img.shape[2]))), axis = 1)
        
    if origin_size[2] > size_z:
        rand_z = random.randint(0, origin_size[2] - size_z)
        img = img[:,:,rand_z:rand_z +size_z]
    else:
        pad_len = size_z - origin_size[1] 
        pad_len_front = random.randint(0, pad_len) 
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((1.337*np.ones((img.shape[0],img.shape[1],pad_len_front)),
            img, 1.337*np.ones((img.shape[0],img.shape[1],pad_len_back))), axis = 2)
    
    
    return img


def center_crop_3d(img, target_shape=crop_shape, **kwargs):
    origin_size = img.shape
    
    if origin_size[0] > target_shape[0]:
        rand_x = origin_size[0]//2-target_shape[0]//2
        img = img[rand_x:rand_x + target_shape[0],:,:]
    else:
        pad_len = target_shape[0] - origin_size[0] 
        pad_len_front = pad_len//2
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((1.337*np.ones((pad_len_front,img.shape[1],img.shape[2])),
            img, 1.337*np.ones((pad_len_back,img.shape[1],img.shape[2]))), axis = 0)

    if origin_size[1] > target_shape[1]:
        rand_y = origin_size[1]//2-target_shape[1]//2
        img = img[:,rand_y:rand_y + target_shape[1],:]
    else:
        pad_len = target_shape[1] - origin_size[1] 
        pad_len_front = pad_len//2
        pad_len_back =  pad_len - pad_len_front
        img = np.concatenate((1.337*np.ones((img.shape[0],pad_len_front,img.shape[2])),
            img, 1.337*np.ones((img.shape[0],pad_len_back,img.shape[2]))), axis = 1)
    
    rand_z = 0
    img = img[:,:,rand_z:rand_z +size_z]  # minimum 24 is guaranteed anyways
    # if origin_size[2] > size_z:
    #     rand_z = origin_size[1]//2-target_shape[1]//2
    #     img = img[:,:,rand_z:rand_z +size_z]
    # else:
    #     pad_len = size_z - origin_size[1] 
    #     pad_len_front = random.randint(0, pad_len) 
    #     pad_len_back =  pad_len - pad_len_front
    #     img = np.concatenate((1.337*np.ones((img.shape[0],img.shape[1],pad_len_front)),
    #         img, 1.337*np.ones((img.shape[0],img.shape[1],pad_len_back))), axis = 2)
        
    return img


# def random_crop_3d(img, target_shape=crop_shape, **kwargs):
#     origin_size = img.shape
#     rand_x = random.randint(0, origin_size[0] - target_shape[0])
#     rand_y = random.randint(0, origin_size[1] - target_shape[1])

#     z = origin_size[2] // 2  # += size_z
#     img = img[rand_x:rand_x + target_shape[0], rand_y:rand_y + target_shape[1], z - size_z:z + size_z]  # 75 +- size_z
#     return img


# def center_crop_3d(img, target_shape=crop_shape, **kwargs):
#     origin_size = img.shape
#     middle = origin_size[0] // 2
#     half = target_shape[0] // 2

#     z = origin_size[2] // 2  # += size_z
#     img = img[middle - half:middle + half, middle - half:middle + half, z - size_z:z + size_z]
#     return img


def gaussian_3d(img, **kwargs):
    sigma = random.uniform(0.01, 0.05)
    noise = np.random.normal(0, sigma, size=img.shape)

    if isinstance(img, list):
        return [i + noise for i in img]

    return img + noise


def flipud_3d(img, **kwargs):
    rand = random.randint(0, 1)
    if isinstance(img, list):
        if rand == 0:
            return [i[::-1, :, :].copy() for i in img]
        else:
            return img

    if rand == 0:
        return img[::-1, :, :].copy()
    else:
        return img


def fliplr_3d(img, **kwargs):
    rand = random.randint(0, 1)
    if isinstance(img, list):
        if rand == 0:
            return [i[:, ::-1, :].copy() for i in img]
        else:
            return img

    if rand == 0:
        return img[:, ::-1, :].copy()
    else:
        return img


# https://github.com/scipy/scipy/issues/5925
def rotate_3d(img, **kwargs):
    angle = random.randrange(0, 360, 45)
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


def channel_fromz(input, **kwargs):#,label = None
    input = np.swapaxes(np.swapaxes(input,0,2),1,2)
    # if label is not None:
    #     label = np.swapaxes(np.swapaxes(label,0,2),1,2)
    return input#, label

def channel_single(input, **kwargs):#,label = None
    input = np.expand_dims(input, 0)
    #if label is not None:
    #    label = np.expand_dims(label, 0)
    return input#, label


def to_tensor(img, **kwargs):
    if isinstance(img, list):
        img = np.stack(img, axis=0)

    #import pdb; pdb.set_trace()
    out = torch.from_numpy(img).float()
    #if len(out.shape) == 3:
    #    out = out.unsqueeze(0)
    return out


# (1, 1), (5, 2), (1, 0.5), (1, 3)
def elastic_transform(img, alpha=0, sigma=0, random_state=None, **kwargs):

    param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]
    if alpha == 0 and sigma == 0:
        rand = random.randint(0, 3)
        alpha, sigma = param_list[rand]

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = crop_shape
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

    trasform_img = np.zeros(img.shape)
    for i in range(img.shape[2]):
        trasform_img[:, :, i] = map_coordinates(img[:, :, i], indices, order=1).reshape(shape)

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


TRAIN_AUGS_3D = [
    random_crop_3d,
    # center_crop_3d,
    # multi_crop_3d,
    calibration,
    gaussian_3d,
    elastic_transform,
    flipud_3d,
    fliplr_3d,
    rotate_3d,
    channel_single,
    to_tensor,
    # Cutout(3, 8)
]

TEST_AUGS_3D = [
    center_crop_3d,
    calibration,
    channel_single,
    # center_multi_crop_3d,
    to_tensor
]

TRAIN_AUGS_25D = [
    random_crop_3d,
    # center_crop_3d,
    # multi_crop_3d,
    calibration,
    gaussian_3d,
    elastic_transform,
    flipud_3d,
    fliplr_3d,
    rotate_3d,
    channel_fromz,
    to_tensor,
    # Cutout(3, 8)
]

TEST_AUGS_25D = [
    center_crop_3d,
    calibration,
    channel_fromz,
    # center_multi_crop_3d,
    to_tensor
]
if __name__ == "__main__":
    path = "/home/dhryu/000_AML_APL_cell_line/dataset/03_adjusted/good/AML_Kasumi/20180820.170521.690.AML_Kasumi1-001.TCF.mat"

    import scipy.io as io

    mat = io.loadmat(path,verify_compressed_data_integrity =False)
    img, ri = mat2npy(mat)
    for t in TRAIN_AUGS_3D:
        img = t(img, ri=ri)
        try:
            print(t, img.shape)
        except:
            print(t, len(img))
