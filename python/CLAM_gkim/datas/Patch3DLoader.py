import os
import random
import h5py

import numpy as np
import scipy.io as io

import torch
from torch.utils import data
from .preprocess3d import TRAIN_NOAUGS_3D

import pathlib

def find_classes(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx # classes: list of classname strings, class_to_idx: a dictionary that links each classname to an int indice


def make_dataset(path, class_to_idx):
    images = []
    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(item)
    return images # a list of tuple (1st tuple element file path, 2nd tuple element classname string)


def mat2npy(mat, **kwargs):
    img = mat['data']
    ri = 1.3374
    return img, ri


class Patch3DSet(data.Dataset):
    def __init__(self, dataset_path, transform=None, aug_rate=0):
        classes, class_to_idx = find_classes(dataset_path)
        print(class_to_idx)
        self.imgs = make_dataset(dataset_path, class_to_idx)

        self.origin_imgs = len(self.imgs)
        if len(self.imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + dataset_path))

        print("Dataset Dir : ", dataset_path, "len : ", len(self.imgs))

        self.augs = TRAIN_NOAUGS_3D if transform is None else transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.aug_rate = aug_rate

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        if pathlib.Path(path).suffix == '.mat':
            mat = io.loadmat(path)
            img, ri = mat2npy(mat)
            img = data['/data']

        elif pathlib.Path(path).suffix == '.h5':
            data = h5py.File(path, 'r')
            img = data['ri'] #data.get('ri').value
        elif pathlib.Path(path).suffix == '.hdf5':
            data = h5py.File(path, 'r')
            img = data['ri'] 
        elif pathlib.Path(path).suffix == '.TCF':
            data = h5py.File(path, 'r')
            img = data['ri'] 
        else:
            raise NameError('file suffix is not h5 nor mat')
        ri = 13374.
        #img = np.array(img).astype('float32')
        img = np.array(img).astype('float32').swapaxes(0,2)
        #print(path)
        #print(img.shape)

        if np.random.uniform(low=0.0,high=1.0,size = 1)[0] <= self.aug_rate:
            # print('augmentation in progress')
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            # print('augmentation not in progress')
            for t in TRAIN_NOAUGS_3D:
                img = t(img, ri=ri)

        """
        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            for t in TEST_AUGS_3D:
                img = t(img, ri=ri)
        """
        #print('img size is {}'.format(img.shape))
        return img, target, path

    def __len__(self):
        return len(self.imgs)


def _make_weighted_sampler(images, nclasses=6): 
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    print(count)
    N = float(sum(count))
    assert N == len(images)
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    return sampler


def Patch3DLoader(image_path, batch_size, sampler=False,
                 transform=None, aug_rate=0,
                 num_workers=1, shuffle=False, drop_last=False):
    dataset = Patch3DSet(image_path, transform=transform, aug_rate=aug_rate)
    if sampler:
        print("Sampler : ", image_path[-5:])
        sampler = _make_weighted_sampler(dataset.imgs)
        return data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


if __name__ == "__main__":
    import torch

    #data_path = "/home/dhryu/000_mice_leukemia/dataset/03_adjusted/split/"
    #data_path = "/data01/dhryu/000_AML_APL_cell_line/dataset/good_1415/"
    data_path = "/data01/dhryu/000_AML_APL_cell_line/dataset/split_binary_circshift_LMJ/"
    import preprocess3d as preprocess

    pp = preprocess.TEST_AUGS_3D
    loader = Patch3DLoader(data_path + "val", 3,
                          transform=pp, aug_rate=0,
                          num_workers=3, shuffle=False, drop_last=False)
    p1 = []
    for input, target, path in loader:
        p1 += list(path)

    p2 = []
    for input, target, path in loader:
        p2 += list(path)

    p3 = []
    for input, target, path in loader:
        p3 += list(path)

    print("3d aug!")
    #cc = len("/data01/dhryu/000_AML_APL_cell_line/dataset/good_1415/") + 1
    cc = len("/data01/dhryu/000_AML_APL_cell_line/dataset/split_binary_circshift_LMJ/")+1
    #cc = len("/data1/BMOL-AI-Data/hycho/human_lymphocyte/smallest/val/") + 1
    for z1, z2, z3 in zip(p1, p2, p3):
        if z1 != z2 or z1 != z3 or z2 != z3:
            print(z1[cc:], z2[cc:], z3[cc:])

