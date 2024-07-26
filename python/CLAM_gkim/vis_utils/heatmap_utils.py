import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from datas.wci_dataset import Wci_Region # from dataset_modules.wsi_dataset import Wsi_Region
#from datas.dataset_h5 import get_eval_transforms
import h5py
from datas.WholeColonyImage import *
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
from utils.constants import MODEL2CONSTANTS
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wci_object=None, vis_level = -1, **kwargs):
    if wci_object is None:
        wci_object = WholeColonyImage(slide_path)
        print(wci_object.name)
    
    colony = wci_object.getColony()

    if vis_level < 0:
        vis_level = colony.get_best_level_for_downsample(1.)#(32)
    heatmap = wci_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wci(wci_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wci_object = WholeColonyImage(wci_path)
    if seg_params['seg_level'] < 0:
        best_level = wci_object.colony.get_best_level_for_downsample(1.)#(32)
        seg_params['seg_level'] = best_level

    wci_object.segmentColony(**seg_params, filter_params=filter_params)
    wci_object.saveSegmentation(seg_mask_path)
    return wci_object

def compute_from_patches(wci_object, img_transforms, feature_extractor=None, clam_pred=None, model=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wci_kwargs):    
    top_left = wci_kwargs['top_left']
    bot_right = wci_kwargs['bot_right']
    patch_size = wci_kwargs['patch_size'] 
    
    from datas.dataset_h5 import Whole_Colony_Bag_FP
    roi_dataset = Wci_Region(wci_object, t=img_transforms, **wci_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    
    #### the below are just test lines for checking if dataset or loader is bad
    #import pdb; pdb.set_trace()
    #roi_dataset2 = Whole_Colony_Bag_FP("/data02/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA_bags/patches/test/00_RA_12h/230425.173926.H9_12hour_treat.014.Group1.A1.S014.h5",wci_object.colony, target_patch_size = 384)
    #roi_loader2 = get_simple_loader(roi_dataset2, batch_size=batch_size, num_workers=8)
    #idx2, (roi2, coords2) = next(enumerate(roi_loader2))
    #roi2 = roi2.to(device)
    #coords2 = coords2.numpy()
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        coords = coords.numpy()
        
        
        with torch.inference_mode():
            
            if isinstance(feature_extractor, nn.DataParallel): ## newly eddited by gkim
                features = feature_extractor.module.encode(roi)
            else:
                features = feature_extractor.encode(roi)
            # _, features = feature_extractor(roi) # edited by GKim as AE3D outputs the decoded result too
            if attn_save_path is not None:
                A = model(features, attention_only=True)
           
                if A.size(0) > 1: #CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(-1, 1).cpu().numpy()
                if False: #ref_scores is not None: removed momentarily by gkim, as the score2percentile function keeps yielding vectors instead of values
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wci_object