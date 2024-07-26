import os
import random
import h5py
from collections import deque
import math
import numpy as np
import scipy.io as io

import torch
from torch.utils import data

from datas.preprocess3d import TEST_AUGS_3D,mat2npy
import time

import pathlib

from datas.TomoLoader import assign_same_values_for_similar_prefixes, find_classes, make_Tomodatalist


class pdset_h5(data.IterableDataset): #IterableDataset not supported for 0.4.1
    def __init__(self, path_img, idx_cls, patch_size, pad_size, rate_overlap, reset_class, transform = [], test =  True):
        self.patch_size = patch_size
        self.pad_size = pad_size
        self.path_img = path_img
        self.idx_cls = idx_cls

            
        self.test = test
        self.transform = transform
        self.rate_overlap = rate_overlap
        self._patch_offset_generation()
        self.reset_class = reset_class
        self.idx_patch = 0
    
    def readimg_h5(self):
        #print(self.path_img)
        label = self.idx_cls
        h5data = h5py.File(self.path_img, 'r')
        for i in h5data.keys():
            if i in ['ri', 'riaif']:
                key_input = i
            elif i in ['bf', 'h&e', 'ihc', 'fl']:
                key_label = i
                label = np.array(h5data[key_label])
        input = np.array(h5data[key_input])
        


        if len(input.shape) > 2:
            input = np.swapaxes(input, 0, 2)
        else:
            input = np.expand_dims(np.swapaxes(input, 0, 1),2)

        
        if abs(13370-(np.mean(input[:20,:20])+np.mean(input[-20:,-20:])+np.mean(input[:20,-20:])+np.mean(input[-20:,:20]))/4) < 200:
            input = np.round(input).astype(np.uint16)
        
        # import pdb;pdb.set_trace()
        if input.shape[0] == 0:
            print(input.shape)
            import pdb; pdb.set_trace()
        return input, label
    
    def __len__(self):
        return self.len
            
    def __getitem__(self, idx = None):
        
        input, label = self.readimg_h5()
        if idx is None:
            idx = self.idx_patch
            c_x, c_y, c_z = self.patch_offset[idx]
            input = input[max(c_x,0): min(c_x + self.patch_size[0],input.shape[0]),
                            max(c_y,0): min(c_y + self.patch_size[1], input.shape[1]),
                            max(c_z,0): min(c_z+self.patch_size[2], input.shape[2])]
            #label = self.label
            self.idx_patch += 1
            for t in self.transform:
                #print(t)
                input = t(input)
        else:
            c_x, c_y, c_z = self.patch_offset[idx]
            input = input[max(c_x,0): min(c_x + self.patch_size[0],input.shape[0]),
                            max(c_y,0): min(c_y + self.patch_size[1], input.shape[1]),
                            max(c_z,0): min(c_z+self.patch_size[2], input.shape[2])]
            #label = self.label
            for t in self.transform:
                #print(t)
                input = t(input)
            
        #import pdb;pdb.set_trace()
        return input, label, self.path_img, [c_x, c_y, c_z]

    def __iter__(self):
        
        input0, label = self.readimg_h5()
        self.idx_iter = 0
        # print('running iter')
       
        if len(input0.shape) > 2: # 3D image
            for c_x, c_y, c_z in self.patch_offset:
                #if max(c_x,0) > min(c_x + self.patch_size[0],input.shape[0]):
                #    print(self.patch_offset)
                #    print(self.path_img)
                #    import pdb; pdb.set_trace()
                input = input0[max(c_x,0): min(c_x + self.patch_size[0],input0.shape[0]),
                            max(c_y,0): min(c_y + self.patch_size[1], input0.shape[1]),
                            max(c_z,0): min(c_z+self.patch_size[2], input0.shape[2])]
                # print([max(c_x,0), min(c_x + self.patch_size[0],input0.shape[0]), max(c_y,0), min(c_y + self.patch_size[1], input0.shape[1]), max(c_z,0), min(c_z+self.patch_size[2], input0.shape[2])])
                # print(input.shape)
                #label = self.label
                if self.reset_class:
                    label = 0
                for t in self.transform:
                    #print(t)
                    input = t(input)
                
                self.idx_iter += 1
                # coord = np.array([c_y, c_x])
                
                #import pdb;pdb.set_trace()
                yield input, label, self.path_img, [c_x, c_y, c_z]#_np2tt(input_np), _np2tt(target_np), int(self.root.split('/')[-2]), self.path
        else: # 2D multi-channel image
            for c_y, c_x in self.patch_offset:
                input = input[:,
                            c_x: c_x + self.patch_size,
                            c_y: c_y + self.patch_size]
                #label = self.label
                if self.reset_class:
                    label = 0
                
                for t in self.transform:
                    input = t(input)
                
                self.idx_iter += 1
                # coord = np.array([c_y, c_x])
                #import pdb;pdb.set_trace()
                yield input, label, self.path_img, [c_x, c_y] #_np2tt(input_np), _np2tt(target_np), int(self.root.split('/')[-2]), self.path
        
        if not self.test:
           self._patch_offset_generation() #reshuffle in case of training

    def _patch_offset_generation(self):
        input, label = self.readimg_h5()
        img_shape = input.shape
        if len(img_shape) > 2:
            x, y, z = img_shape
        else:
            x, y = img_shape
        x = max(x, self.pad_size)
        y = max(y, self.pad_size)
        x_n = self.patch_size[0] // self.rate_overlap # step size 
        y_n = self.patch_size[1] // self.rate_overlap
        range_h =  math.ceil((y - self.patch_size[1]) / y_n) + 1
        range_w = math.ceil((x - self.patch_size[0]) / x_n) + 1
        if len(img_shape) > 2:
            if self.patch_size[2] > self.rate_overlap:
                z_n = self.patch_size[2] // self.rate_overlap
            else:
                z_n = self.patch_size[2]
            range_d = math.ceil((z - self.patch_size[2]) / z_n) + 1

        self.patch_offset = deque()
        for h in range(range_h):
            for w in range(range_w):

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
                y_offset = int(y_offset)
                x_offset = int(x_offset)

                if y_offset < 0:
                    y_offset = 0

                if len(img_shape) > 2:
                    for d in range(range_d):
                        if self.test:
                            z_offset = z_n * d
                        else:
                            z_offset = random.randint(0,math.ceil((z - self.patch_size[2]) / z_n))
                        if z_offset + self.patch_size[2] > z:
                            z_offset = z - self.patch_size[2]
                        z_offset = int(z_offset)
                        self.patch_offset.append((x_offset, y_offset, z_offset))
                else:
                    self.patch_offset.append((x_offset, y_offset))
        self.len = len(self.patch_offset)
        # print('shape for {} : {}'.format(self.path_img,img_shape))
        # print('patch_offset for {} : {}'.format(self.path_img,self.patch_offset))

def getdatasets(path, patch_size, pad_size, rate_overlap, transform = [], test = True, aug_rate = 0,
                pats_exclude = (),pats_class = (),
                reset_class = False, mode_class = 1):
    classes, class_to_idx = find_classes(path, pats_class, mode_class)
    print(class_to_idx)
    datalist = make_Tomodatalist(path, class_to_idx, pats_exclude, pats_class)
    #if len(datalist) == 0:
    #    raise (RuntimeError("Found 0 images in " + path))
    print("Dataset Dir : ", path, "len : ", len(datalist))
    if aug_rate != 0:
        datalist += random.sample(datalist, int(len(datalist) * aug_rate))
    transform = transform
    classes = classes
    class_to_idx = class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    datasets = []
    for path_img, idx_cls in datalist:
        datasets.append(pdset_h5(path_img, idx_cls, patch_size, pad_size, rate_overlap, reset_class, transform, test))
    for pdset in datasets:
        pdset.classes = classes
        pdset.class_to_idx = class_to_idx
    return datasets, classes, class_to_idx
        