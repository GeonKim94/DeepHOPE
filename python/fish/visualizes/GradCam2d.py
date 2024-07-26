# Author : Daewoong Ahn
import copy

import torch
import torch.nn.functional as F
from torch.autograd import Function

#mport cv2
import numpy as np
from skimage.transform import resize as skresize
from torchvision.transforms import Resize as TorchVisionResize



class GradCam2d:
    def __init__(self, model, name_part, idx_layer, device):#(self, model, hooks, device):
        #if isinstance(hooks, list) is False:
        #    raise ValueError("Hooks must be list of str")
        
        self.device = device
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model = self.model.to(device)

        self.feature = []
        self.grad = []
        #print(self.model._modules.keys())

        # Setting Hook
#        hook_layer = self.model
#        for h in hooks:
#            #import pdb; pdb.set_trace()
#            if isinstance(h, str) is False:
#                raise ValueError("Hooks must be list of str")
#            if h not in hook_layer._modules.keys():
#                raise ValueError("Hook[{}] is not in {}".format(h, hook_layer._modules.keys()))
#            hook_layer = hook_layer._modules[h]

        #import pdb;pdb.set_trace()
        #print(10)
        #hook_layer = self.model._modules['head_layer'][0]._modules['transfer']._modules['layers'][1]._modules['layers'][2]
        #hook_layer = self.model._modules['stem'][0]
        #hook_layer = self.model._modules['tail_layer'][0]._modules['layer']._modules['layer'][0]
        #hook_layer = self.model._modules['tail_layer'][1]._modules['layer']._modules['layer'][0]
        #hook_layer = self.model._modules['body_layer'][2]._modules['layer']._modules['layer'][3]
        #hook_layer = self.model._modules['head_layer'][0]._modules['layer'][0]._modules['layers'][0]
        #import pdb;pdb.set_trace()
        hook_layer = self.model._modules[name_part][idx_layer]._modules['layer']._modules['layer'][0]

        def save_feature(module, input_, output):
            self.feature.append(output.detach())

        def save_grad(module, grad_in, grad_out):
            self.grad.append(grad_out[0].detach())

        self.grad_hook = hook_layer.register_backward_hook(save_grad)
        self.feature_hook = hook_layer.register_forward_hook(save_feature)
        
        

    @staticmethod
    def get_layer_name(model):
        for n, m in model._modules.items():
            print("Name : ", n, "\nModule : ", m)

    def remove_hook_data(self):
        self.feature = []
        self.grad = []

#    def backward(self, output):
#        _, idx = output.max(dim=1)
#        one_hot = torch.zeros_like(output).to(self.device)
#        one_hot[0][idx] = 1
#        self.model.zero_grad()
#        output.backward(gradient=one_hot, retain_graph=True)
    
    def backward(self, output, idx):
        one_hot = torch.zeros_like(output).to(self.device)
        one_hot[0][idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
    
    
    def gunho(self, grad, feature):
        alpha = torch.sum(grad, dim=(1, 2, 3), keepdim=True)
        gcam = (alpha * grad).sum(dim=0)
        gcam = gcam.detach().cpu().numpy()
        return gcam

    def gdcam(self, grad, feature):
        grad = torch.unsqueeze(grad, 0)
        weights = F.adaptive_avg_pool2d(grad, 1)
        weights.squeeze_(0)

        gcam = (feature * weights).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)
        gcam = gcam.detach().cpu().numpy()
        return gcam
    
    def gdcampp(self, grad, feature):
        grad_2 = grad * grad
        grad_3 = grad * grad_2
        weights = grad_2 / ((2 * grad_2) + (grad_3 * feature) + 1e-12)
        
        weights.unsqueeze_(0)
        weights = F.adaptive_avg_pool3d(weights, 1)
        weights.squeeze_(0)
        
        feature = torch.clamp(feature, min=0.)
        gcam = (feature * weights).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)
        gcam = gcam.detach().cpu().numpy()
        return gcam
    
    def respondcam(self, grad, feature):
        weights = (grad * feature).sum(dim=(1,2)) / (feature.sum(dim=(1,2)) + 1e-8)
        weights.unsqueeze_(1).unsqueeze_(2)#.unsqueeze_(3)        
        gcam = (feature * weights).sum(dim=0)
        gcam = torch.abs(gcam)
        gcam = torch.sigmoid(gcam)
        gcam = gcam.detach().cpu().numpy()
        return gcam
    

    def __call__(self, input_, idx=None, mode="gunho"):
        with torch.no_grad():
            output = self.model(input_.to(self.device))[0]
        _, pred_idx = output.max(dim=1)
        if idx == None:
            idx = pred_idx
        self.backward(output,idx)

        # idx : [-1] : last output, [0] : remove batch dim
        feature_map = self.feature[-1][0]
        grad = self.grad[-1][0]
        if mode == "gunho":
            gcam = self.gunho(grad, feature_map)
        elif mode == "gdcam":
            gcam = self.gdcam(grad, feature_map)
        elif mode == "gdcampp":
            gcam = self.gdcampp(grad, feature_map)
        elif mode == "respond":
            gcam = self.respondcam(grad, feature_map)
            
        gcam -= gcam.min() ; gcam /= gcam.max()
        gcam = skresize(gcam, input_.shape[-2:], mode="constant", anti_aliasing=True)
        gcam -= gcam.min() ; gcam /= gcam.max()
        return gcam, pred_idx
    
    def multicam(self, input_, idx=None):
        output = self.model(input_.to(self.device))[0]
        #print(output)
        _, pred_idx = output.max(dim=1)
        if idx == None:
            idx = pred_idx
        self.backward(output,idx)

        # idx : [-1] : last output, [0] : remove batch dim
        feature_map = self.feature[-1][0]
        grad = self.grad[-1][0]
        #gunho = self.gunho(grad, feature_map)
        gd = self.gdcam(grad, feature_map)
        gdpp = self.gdcampp(grad, feature_map)
        resp = self.respondcam(grad, feature_map)
        def _resize(gcam):
        

            #gcam -= gcam.min() ; gcam /= gcam.max()
            #gcam = skresize(gcam, input_.shape[-3:], mode="constant", anti_aliasing=True)
            #gcam -= gcam.min() ; gcam /= gcam.max()
            return gcam
        
        resp = _resize(resp)
        gd = _resize(gd)
        gdpp = _resize(gdpp)
        #gunho = _resize(gunho)
        return resp, gd, gdpp, pred_idx#resp, gd, gdpp, gunho, pred_idx
    @staticmethod
    def cam_on_image(img, cam_mask):
        pass
    