import os
import random
import h5py

import numpy as np
import scipy.io as io

import torch
from torch.utils import data
import preprocess3d
import math
import pathlib

def find_classes(path):
    classes = sorted([d for d in next(os.walk(path))[1]]) # d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    if len(classes) == 0:
        classes = None
        classes_to_idx = None
    return classes, class_to_idx


def make_h5datalist(path): # list of data that has the data path and the class info (common for all)
    images = []
    path = os.path.expanduser(path)

    if len(next(os.walk(path))[1]) > 1:
        classes, class_to_idx = find_classes(path)
        for cls in next(os.walk(path))[1]:
            d = os.path.join(path, cls)
            if not os.path.isdir(d):
                continue

            root, folders, fnames = next(os.walk(d))
            for fname in sorted(fnames):
                if fname.endswith('h5'):
                    path_data = os.path.join(root, fname)
                    item = (path_data, class_to_idx[cls])
                    images.append(item)
    else: #when there is no class
        root, folders, fnames = next(os.walk(path))
        for fname in sorted(fnames):
            if fname.endswith('h5'):
                path_data = os.path.join(root, fname)
                item = (path_data, None)
                images.append(item)
    return images


class pdset_h5(data.IterableDataset):
    def __init__(self, path_img, patch_size, transform = [], test =  True):
        self.patch_size = patch_size
        self.path_img = path_img
        h5data = h5py.File(path_img, 'r')
        for i in h5data.keys():
            if i in ['ri', 'riaif']:
                key_input = i
            elif i in ['bf', 'h&e', 'ihc', 'fl']:
                key_label = i
        self.input = np.array(h5data[key_input])
        self.input = np.swapaxes(self.input, 0, 2)
        self.label = np.array(h5data[key_label])
        self.label = np.swapaxes(self.label, 0, 2)
        self.test = test
        self.transform = transform
        self._patch_offset_generation()

    def __iter__(self):
        self.idx = 0
        #import pdb; pdb.set_trace()
        if len(self.input.shape) > 2:
            for c_z, c_y, c_x in self.patch_offset:
                input = self.input[c_x: c_x + self.patch_size[0],
                            c_y: c_y + self.patch_size[1],
                            c_z: c_z+self.patch_size[2]]
                label = self.label[c_x: c_x + self.patch_size[0],
                            c_y: c_y + self.patch_size[1],
                            c_z: c_z+self.patch_size[2]]

                for t in self.transform:
                    #print(t)
                    input, label = t(input, label)
                
                self.idx += 1
                # coord = np.array([c_y, c_x])
                yield input, label, self.path_img, [c_x, c_y, c_z]#_np2tt(input_np), _np2tt(target_np), int(self.root.split('/')[-2]), self.path
        else:
            for c_y, c_x in self.patch_offset:
                input = self.input[:,
                            c_x: c_x + self.patch_size,
                            c_y: c_y + self.patch_size]
                label = self.label[:,
                                    c_x: c_x + self.patch_size,
                                    c_y: c_y + self.patch_size]
                
                for t in self.transform:
                    input, label = t(input, label)
                
                self.idx += 1
                # coord = np.array([c_y, c_x])
                yield input, label, self.path_img, [c_x, c_y] #_np2tt(input_np), _np2tt(target_np), int(self.root.split('/')[-2]), self.path
        
        if not self.test:
           self._patch_offset_generation() #reshuffle in case of training

    def _patch_offset_generation(self):
        #import pdb; pdb.set_trace()
        img_shape = self.input.shape
        if len(img_shape) > 2:
            x, y, z = img_shape
        else:
            x, y = img_shape
        x_n = self.patch_size[0] // 2 # stride overlap ratio = 1/2
        y_n = self.patch_size[1] // 2
        if len(img_shape) > 2:
            z_n = self.patch_size[2] // 2

        self.patch_offset = deque()
        for h in range(math.ceil((y - self.patch_size[1]) / y_n) + 1):
            for w in range(math.ceil((x - self.patch_size[0]) / x_n) + 1):

                if self.test:
                    y_offset = y_n * h
                    x_offset = x_n * w
                else:
                    y_offset = random.randint(0,math.ceil((y - self.patch_size[1]) / y_n))
                    x_offset = random.randint(0,math.ceil((x - self.patch_size[0]) / x_n))

                if y_offset + self.patch_size[1] > y:
                    y_offset = y - self.patch_size[1]
                if x_offset + self.patch_size[0] > x:
                    x_offset = x - self.patch_size[0]

                if len(img_shape) > 2:
                    for d in range(math.ceil((z - self.patch_size[2]) / z_n) + 1):
                        if self.test:
                            z_offset = z_n * d
                        else:
                            z_offset = random.randint(0,math.ceil((z - self.patch_size[2]) / z_n))
                        if z_offset + self.patch_size[2] > z:
                            z_offset = z - self.patch_size[2]
                        self.patch_offset.append((z_offset, y_offset, x_offset))
                else:
                    self.patch_offset.append((y_offset, x_offset))
        self.len = len(self.patch_offset)

class cpdset_h5(data.ChainDataset):

    def __iter__(self):
        worker_info = data.get_worker_info()
        worker_id = worker_info.id
        for d in self.datasets[worker_id]:
            assert isinstance(d, data.IterableDataset), \
                "ChainDataset only supports IterableDataset"
            for x in d:
                yield x


def getdatasets(path, patch_size, transform = [], test = True, aug_rate = 0):
    classes, class_to_idx = find_classes(path)
    print(class_to_idx)
    datalist = make_h5datalist(path)
    if len(datalist) == 0:
        raise (RuntimeError("Found 0 images in " + path))
    print("Dataset Dir : ", path, "len : ", len(datalist))
    if aug_rate != 0:
        datalist += random.sample(datalist, int(len(datalist) * aug_rate))
    transform = transform
    classes = classes
    class_to_idx = class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    datasets = []
    for path_img, idx_cls in datalist:
        datasets.append(pdset_h5(path_img, patch_size, transform, test))
    return datasets
        

