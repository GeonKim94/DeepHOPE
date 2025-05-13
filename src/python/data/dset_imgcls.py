import torch
from torch.utils.data import Dataset
from .utils_data import readimg

from pathlib import Path

import os
from os.path import basename, splitext, isfile, isdir, exists
import yaml


class dset_imgcls(Dataset):
    def __init__(self, dataset_path, preps = None, reset_class = False, mode_class = "binary"):
        super().__init__()
        self.read_dataset(dataset_path, mode_class)
        self.reset_class = reset_class
        self.preps = preps

    def __len__(self):
        return len(self.classes_data)
        
    def read_dataset(self,dataset_path, mode_class):
        if not exists(dataset_path):
            self.paths_data = []
            self.classes_data = []
            self.classnames = []
            pass
        if isfile(dataset_path):
            if splitext(dataset_path)[-1] == '.yaml':
                self.paths_data, self.classes_data, self.classnames = extract_yaml(dataset_path)
            else:
                raise ValueError('Data list format not supported.')
        elif isdir(dataset_path):
            self.paths_data, self.classes_data, self.classnames = extract_dir(dataset_path)
        if mode_class == "binary":
            self.classes_data = [1 if 'ctl' in class_data else 0 for class_data in self.classes_data]
            self.classnames = ['05_ctl', '99_other']
        else:
            e2i = {value: idx for idx, value in enumerate(self.classnames)}
            self.classes_data = [e2i[ele] for ele in self.classes_data]

    def __getitem__(self, idx):
        path_ = self.paths_data[idx]
        if self.classes_data is None:
            target_ = 0
        else:
            target_ = self.classes_data[idx]

        if self.reset_class:
            target_ = 0
        target_ = torch.tensor(target_, dtype=torch.int64)
        data_ = readimg(path_)
        for p in self.preps:
            data_ = p(data_)
        return data_, target_, path_


def extract_dir(dataset_path):
    paths_data_ = []
    classes_data_ = []
    classnames_ = []
    for class_, name_class_ in enumerate(sorted(os.listdir(dataset_path))):
        path_class = os.path.join(dataset_path,name_class_)
        if not isdir(path_class):
            continue
        classnames_.append(name_class_)
        list_file = os.listdir(path_class)
        for fname in list_file:
            if not isfile(os.path.join(path_class, fname)):
                continue
            paths_data_.append(os.path.join(path_class, fname))
            classes_data_.append(name_class_)

    return paths_data_, classes_data_, classnames_
        
def extract_yaml(dataset_path):
    if splitext(basename(dataset_path))[-1] == '.yaml':
        dict_yaml = yaml.safe_load(Path(dataset_path).read_text())
    else:
        raise ValueError('The dataset path is not a yaml file. Check file format.')
    paths_data_ = dict_yaml['paths']
    if 'classnames' in dict_yaml.keys():
        classnames_ = dict_yaml['classes']
    else:
        classnames_ = None
    if "classes_data" in dict_yaml.keys():
        classes_data_ = dict_yaml['classes_data']
        print('Class info is given')
    else:
        classes_data_ = None
        print('Class info is unknown')
    return paths_data_, classes_data_, classnames_