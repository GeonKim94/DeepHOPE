# Author : Daewoong Ahn

import torch
import torch.nn.functional as F
from torch.autograd import Function

from torchvision import models
from torchvision import utils

#import cv2
import sys
import numpy as np
import argparse
import copy
import scipy
from skimage.transform import resize as skresize
    
class RespondCam:
    def __init__(self, model, hooks, device):
        if isinstance(hooks, list) is False:
            raise ValueError("Hooks must be list of str")
        
        self.device = device
        self.model = copy.deepcopy(model)
        self.model.eval()

        self.feature = []
        self.grad = []

        # Setting Hook
        hook_layer = self.model
        for h in hooks:
            if isinstance(h, str) is False:
                raise ValueError("Hooks must be list of str")
            if h not in hook_layer._modules.keys():
                raise ValueError("Hook[{}] is not in {}".format(h, hook_layer._modules.keys()))
            hook_layer = hook_layer._modules[h]

        print("Hook Layer : ", hook_layer)
        self.back_hook = hook_layer.register_backward_hook(self.hook_save_grad)
        self.feature_hook = hook_layer.register_forward_hook(self.hook_save_feature)

    @staticmethod
    def get_layer_name(model):
        for n, m in model._modules.items():
            print("Name : ", n, "\nModule : ", m)

    def hook_save_feature(self, layer, input_, output):
        self.feature.append(output.detach())

    def hook_save_grad(self, layer, grad_in, grad_out):
        self.grad.append(grad_out[0].detach())

    def remove_hook_data(self):
        self.feature = []
        self.grad = []

    def backward(self, output):
        value, idx = output.max(dim=1)
        one_hot = torch.zeros_like(output).to(self.device)
        one_hot[0][idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

    def __call__(self, input_, index=None):
        if len(input_.shape) != 5:
            raise ValueError("Input must be [B C H W Z] shape(3d image input)")

        self.remove_hook_data()

        output = self.model(input_.to(self.device))
        if isinstance(output, tuple):
            output = output[0]
    
        self.backward(output)

        grad = self.grad[0].to(self.device)[0]
        feature = self.feature[0].to(self.device)[0]

        weights = (grad * feature).sum(dim=(1,2,3)) / feature.sum(dim=(1,2,3))
        weights.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)

        cam = (feature * weights).sum(dim=0)
        print("Cam minmax : ", cam.min(), cam.max())
        # cam.clamp_(min=0.)
        
        cam = cam.detach().cpu().numpy()
        # normalize for sklearn resize
        cam -= cam.min() ; cam /= cam.max()
        # cam = cv2.resize(cam, input_.shape[-3:])
        # cam = scipy.misc.imresize(cam, input_.shape[-3:], interp="lanczos")
        cam = skresize(cam, input_.shape[-3:], mode="reflect", anti_aliasing=True)
        
        cam = 2 * (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) - 1
        self.remove_hook_data()
        return cam, output, feature, grad

    '''
    @staticmethod
    def cam_on_image(img, mask):
        # TODO: FIX!!
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HOT)
        heatmap = np.float32(heatmap) / 255
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        cam = np.multiply(heatmap, np.float32(img))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    '''

def _preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input_ = preprocessed_img
    input_.requires_grad_(True)
    return input_


if __name__ == '__main__':
    from GuidedBackpropReLUModel import GuidedBackpropReLUModel
    from datas.BacLoader import bacLoader
    from datas.preprocess3d import TRAIN_AUGS_3D, TEST_AUGS_3D
    from Densenet3d_nomaxpool import d169_3d


    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    device = torch.device("cpu")
    model = d169_3d(num_classes=18, sample_size=64, sample_duration=96)
    grad_cam = RespondCam(model=model, hooks=["features", "denseblock2", "denselayer12", "conv1"], device=device)
    gb_model = GuidedBackpropReLUModel(model=model, device=device)

    data_path = "/data2/DW/180930_bac/valid40/"
    train_loader = bacLoader(data_path + "train", 1, task="bac", sampler=False,
                                transform=TEST_AUGS_3D, aug_rate=0,
                                num_workers=1, shuffle=False, drop_last=True)

    for input_, *_ in train_loader:
        break

    target_index = None
    cam_mask = grad_cam(input_, target_index)
    
    exit()
    # cam_img is cam heatmap on orignal image.
    cam_img = RespondCam.cam_on_image(img, cam_mask)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    gb_img = gb_model(input_, index=target_index)

    gb_mask = np.zeros(gb_img.shape)
    for i in range(0, gb_img.shape[0]):
        gb_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb_img)

    utils.save_image(cam_img, "cam.jpg")
    utils.save_image(torch.from_numpy(gb_img), 'gb.jpg')
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
