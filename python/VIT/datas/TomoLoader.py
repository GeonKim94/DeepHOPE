import os
import random
import h5py

import numpy as np
import scipy.io as io

import torch
from torch.utils import data

from datas.preprocess3d import TEST_AUGS_3D,mat2npy
import time

import pathlib

def assign_same_values_for_similar_prefixes(dictionary):
    prefix_to_idx = {}
    new_dict = {}
    next_idx = 0

    for key in sorted(dictionary.keys()):
        prefix = key[:2]
        if prefix not in prefix_to_idx:
            prefix_to_idx[prefix] = next_idx
            next_idx += 1
        new_dict[key] = prefix_to_idx[prefix]

    return new_dict

def find_classes(path, pats_class, mode_class = 1): #mode_cls 0 treats the folders independently #mode_cls 1 groups folders with first two charactesr

    if len(pats_class) != 0 :
        classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and any(pat in d for pat in pats_class)])
        class_to_idx = {classes[i]: i for i in range(len(classes))}
    else:
        classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        class_to_idx = {classes[i]: i for i in range(len(classes))}

    if mode_class == 1:
        class_to_idx = assign_same_values_for_similar_prefixes(class_to_idx)

    return classes, class_to_idx


def make_Tomodatalist(path, class_to_idx, pats_exclude = (), pats_class = ()):
    # import pdb;pdb.set_trace()
    images = []
    path = os.path.expanduser(path)
    print(pats_exclude)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            print(d)
            continue
        if not any(pat in target for pat in pats_class):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                found = any(pattern in fname for pattern in pats_exclude)
                if found:
                    #print("Skipping {} as it contains exclusion pattern in the filename.".format(fname))
                    # print(fname)
                    continue
                
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(item)
    return images


class TomoSet(data.Dataset):
    def __init__(self, dataset_path, transform=None, aug_rate=0, pats_exclude = (), pats_class = (), reset_class = False, mode_class = 1):
        classes, class_to_idx = find_classes(dataset_path, pats_class, mode_class = mode_class)
        print(class_to_idx)
        self.imgs = make_Tomodatalist(dataset_path, class_to_idx, pats_exclude, pats_class)
        self.reset_class = reset_class

        self.origin_imgs = len(self.imgs)
        # if len(self.imgs) == 0:
        #     raise (RuntimeError("Found 0 images in subfolders of: " + dataset_path))

        print("Dataset Dir : ", dataset_path, "len : ", len(self.imgs))

        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

        self.augs = [] if transform is None else transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # start = time.time()
        path, target = self.imgs[index]

        if self.reset_class: # for inferring totally irrelevant data iPSCs
            target = 0
        #print(path)
        if pathlib.Path(path).suffix == '.mat':
            mat = io.loadmat(path)
            img, ri = mat2npy(mat)
            #img = data['/data']

        elif pathlib.Path(path).suffix == '.h5':
            data = h5py.File(path, 'r') # 220801 for -v7.3 mat files
            img = data.get('/ri')#.value # deprecated
            if np.max(img) > 10000:
                ri = 13370
            else:
                ri = 1.337

            if len(img.shape)>2:
                img = np.swapaxes(img,0,2)
        else:
            raise NameError('file suffix is not h5 nor mat')

        for t in self.augs:
            #print(img.shape)
            img = t(img)#, ri=ri)
        #print(img.shape)

        """
        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            for t in TEST_AUGS_3D:
                img = t(img, ri=ri)
        """
        
        # end = time.time()
        # print("data loaded, time elapsed = {}".format(end - start))
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


def TomoLoader(image_path, batch_size, sampler=False,
                 transform=None, aug_rate=0,
                 num_workers=1, shuffle=False, drop_last=False, pats_exclude = (), pats_class = (), reset_class = False, mode_class = 1):
    dataset = TomoSet(image_path, transform=transform, aug_rate=aug_rate, pats_exclude=pats_exclude, pats_class = pats_class, reset_class = reset_class, mode_class = mode_class)
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
    loader = TomoLoader(data_path + "val", 3,
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

