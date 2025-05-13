import math
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_wide(in_planes, out_planes, kernel_size = 5, rate_overlap = 1):
    "convolution block with given sampling rate"
    pad_front = math.ceil(kernel_size/2-1/2)
    pad_back = math.floor(kernel_size/2-1/2)
    stride = kernel_size/rate_overlap
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding= [pad_front, pad_front, pad_back, pad_back], bias=False)