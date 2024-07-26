#from torchvision import transforms
import pandas as pd
import numpy as np
import time
import pdb
import PIL.Image as Image
import h5py
from torch.utils.data import Dataset
import torch
from utils.utils_classes import Contour_Checking_fn, isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard

def get_contour_check_fn(contour_fn='four_pt_hard', cont=None, ref_patch_size=None, center_shift=None):
    if contour_fn == 'four_pt_hard':
        cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size, center_shift=center_shift)
    elif contour_fn == 'four_pt_easy':
        cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size, center_shift=0.5)
    elif contour_fn == 'center':
        cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size)
    elif contour_fn == 'basic':
        cont_check_fn = isInContourV1(contour=cont)
    else:
        raise NotImplementedError
    return cont_check_fn



class Wci_Region(Dataset): # gkim: trying to fix this bc I need 3D and this givese 2D lol (not sure if I use it for anywhere else)
    '''
    args:
        wci_object: instance of WholeSlideImage wrapper over a WCI
        top_left: tuple of coordinates representing the top left corner of WCI region (Default: None)
        bot_right tuple of coordinates representing the bot right corner of WCI region (Default: None)
        level: downsample level at which to prcess the WCI region
        patch_size: tuple of width, height representing the patch size
        step_size: tuple of w_step, h_step representing the step size
        contour_fn (str): 
            contour checking fn to use
            choice of ['four_pt_hard', 'four_pt_easy', 'center', 'basic'] (Default: 'four_pt_hard')
        t: custom torchvision transformation to apply 
        custom_downsample (int): additional downscale factor to apply 
        use_center_shift: for 'four_pt_hard' contour check, how far out to shift the 4 points
    '''
    def __init__(self, wci_object, top_left=None, bot_right=None, level=0, 
                 patch_size = (384, 384), step_size=(192, 192), 
                 contour_fn='four_pt_hard',
                 t=None, custom_downsample=1, use_center_shift=False):
        
        self.custom_downsample = custom_downsample

        # downscale factor in reference to level 0
        self.ref_downsample = wci_object.level_downsamples[level]
        # patch size in reference to level 0
        self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        if self.custom_downsample > 1:
            self.target_patch_size = patch_size
            patch_size = tuple((np.array(patch_size) * np.array(self.ref_downsample) * custom_downsample).astype(int))
            step_size = tuple((np.array(step_size) * custom_downsample).astype(int))
            self.ref_size = patch_size
        else:
            step_size = tuple((np.array(step_size)).astype(int))
            self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        self.colony = wci_object.colony
        self.level = level
        self.patch_size = patch_size
            
        if not use_center_shift:
            center_shift = 0.
        else:
            overlap = 1 - float(step_size[0] / patch_size[0])
            if overlap < 0.25:
                center_shift = 0.375
            elif overlap >= 0.25 and overlap < 0.75:
                center_shift = 0.5
            elif overlap >=0.75 and overlap < 0.95:
                center_shift = 0.5
            else:
                center_shift = 0.625
            #center_shift = 0.375 # 25% overlap
            #center_shift = 0.625 #50%, 75% overlap
            #center_shift = 1.0 #95% overlap
        
        filtered_coords = []
        #iterate through tissue contours for valid patch coordinates
        for cont_idx, contour in enumerate(wci_object.contours_colony): 
            print('processing {}/{} contours'.format(cont_idx, len(wci_object.contours_colony)))
            cont_check_fn = get_contour_check_fn(contour_fn, contour, self.ref_size[0], center_shift)
            #pdb.set_trace()
            if wci_object.holes_colony is not None:
                contour_holes = wci_object.holes_colony[cont_idx]
            else:
                contour_holes = None
            coord_results, _ = wci_object.process_contour(contour, contour_holes, level, '', # last is save_path (must be '' because not saved)
                            patch_size = patch_size[0], step_size = step_size[0], contour_fn=cont_check_fn,
                            use_padding=True, top_left = top_left, bot_right = bot_right)
            if len(coord_results) > 0:
                filtered_coords.append(coord_results['coords'])
        
        coords=np.vstack(filtered_coords)

        self.coords = coords
        print('filtered a total of {} coordinates'.format(len(self.coords)))
        
        # apply transformation
        assert t is not None, 'transformations not provided'
        self.transforms = t

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        coord = self.coords[idx]
        #patch = self.colony.read_region(tuple(coord), self.level, self.patch_size).convert('RGB')
        patch = self.colony.read_region(tuple(coord), self.level, self.patch_size)
        #patch = np.max(patch, axis = 2)
        # patch = (patch-13300.)/(14000.-13300.)
        # patch[patch<0] = 0
        # patch[patch>1] = 1
        #patch = (patch*255).astype(np.uint8)
        #patch = np.array(Image.fromarray(patch).convert("RGB"))

        if self.custom_downsample > 1:
            patch = patch.resize(self.target_patch_size)
        
        #patch = self.transforms(patch).unsqueeze(0)
        if len(patch.shape) > 2: # 3D patch
            patch = np.swapaxes(patch,0,2)
        else: # 2D patch
            patch = np.swapaxes(patch,0,1)


        for t in self.transforms:
            patch = t(patch, ri=13374.)
        patch = torch.unsqueeze(patch, 0)

        #print(patch.shape)
        return patch, coord 
    