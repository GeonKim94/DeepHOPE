from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
#from models import get_encoder #removed by GKim
from models import AE3D
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from models.AE3D import AE3D_SEC1
from utils.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wci, drawHeatmap, compute_from_patches
from utils.colony_utils import sample_rois#wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from tqdm import tqdm
from PIL import Image
from extract_features_fp import load

from datas.preprocess3d import TEST_AUGS_3D, TRAIN_AUGS_3D


resx = 0.155432865023613
resy = 0.155432865023613
resz = 0.949573814868927

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="config_gkim.yaml")
parser.add_argument('--path_load', type=str, default=None) ### added by gkim (feature extractor)
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
    features = features.to(device)
    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)) or isinstance(model.module, (CLAM_SB, CLAM_MB)):

            model_results_dict = model(features)
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()

            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
        
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params

def parse_config_dict(args, config_dict):

    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

if __name__ == '__main__':
    path_load_FE = args.path_load ## added by gkim
    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n'+key)
            for value_key, value_value in value.items():
                print (value_key + " : " + str(value_value))
        else:
            print ('\n'+key + " : " + str(value))
            
    decision = input('Continue? Y/N ')
    if decision in ['Y', 'y', 'Yes', 'yes']:
        pass
    elif decision in ['N', 'n', 'No', 'NO']:
        exit()
    else:
        raise NotImplementedError

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    encoder_args = args['encoder_arguments']
    encoder_args = argparse.Namespace(**encoder_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])
    
    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    preset = data_args.preset
    def_seg_params = {'seg_level': 0, 'sthresh': 13420, 'sthresh_up': 15000, 'mthresh': 1, 'close': round(2/resx), 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t': 100}#{'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        
        #dict_preset_df = preset_df.to_dict(orient = 'records')[0]
        
        for key in def_seg_params.keys(): # changed so that un-matching dictionaries are ignored
            if key in preset_df.columns.tolist():
                def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            # slides = sorted(os.listdir(data_args.data_dir)) # GKim replace
            	# List to store relative paths of .h5 files
            slides = []
            # Walk through the directory structure
            for root, dirs, files in os.walk(data_args.data_dir):
                # Check if the current directory is a class folder (starts with '0')
                if os.path.basename(root).startswith('0'):
                    # Iterate over files in the current directory
                    for file in files:
                        # Check if the file has a '.h5' extension
                        if file.endswith('.h5'):
                            # Construct the relative path and append to slides
                            relative_path = os.path.relpath(os.path.join(root, file), data_args.data_dir)
                            slides.append(relative_path)
                            #print(relative_path)
            slides.sort()
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
        
    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))
    
    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
        model.eval()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)   
    else:
        raise NotImplementedError

    #feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    feature_extractor = AE3D_SEC1(input_channels=1, latent_dim=model_args.embed_dim)
    img_transforms = TEST_AUGS_3D
    load(feature_extractor, path_load_FE)

    _ = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    if torch.cuda.device_count() > 1:
        feature_extractor = nn.DataParallel(feature_extractor)
    print('Done!')

    label_dict =  data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    ###step size has to be determined here!
    blocky_wci_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': step_size,#tuple([int(i/2) for i in patch_size]), 
    'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

    for i in tqdm(range(len(process_stack))): ### for progress bar of how many colonies (slides) are done
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name+=data_args.slide_ext
        print('\nprocessing: ', slide_name)	

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        print('Initializing wci object')
        wci_object = initialize_wci(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wci_ref_downsample = wci_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(patch_size) * np.array(wci_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wci_object.colony.get_best_level_for_downsample(1.)
            vis_params['vis_level'] = best_level
        mask = wci_object.visWCI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
    
        ##### check if h5_features_file exists ######
        if not os.path.isfile(h5_path):
            _, _, wci_object = compute_from_patches(wci_object=wci_object, 
                                            model=model, 
                                            feature_extractor=feature_extractor, 
                                            img_transforms=img_transforms,
                                            batch_size=exp_args.batch_size, **blocky_wci_kwargs, 
                                            attn_save_path=None, feat_save_path=h5_path, 
                                            ref_scores=None)		
        
        ##### check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            file.close()

        # load features 
        features = torch.load(features_path)
        process_stack.loc[i, 'bag_size'] = len(features)
        
        wci_object.saveSegmentation(mask_file)
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)

        # del features # removed by gkim for debugging
        
        if not os.path.isfile(block_map_save_path): 
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        # save top 3 predictions
        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
        else:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
        
        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()
        

        samples = sample_args.samples

        for sample in samples: # gkim note: i don't see why there is a for loop... it's only a single dictionary...
            if sample['sample']:
                tag = "label_{}_pred_{}".format(label, Y_hats[0])
                # sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
                sample_save_dir = os.path.join('/data02/gkim/stem_cell_jwshin/patch_save/', str(tag), sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    #patch = wci_object.colony.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch = wci_object.colony.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size))
                    patch = np.max(patch, axis = 2)
                    patch = (patch-13300.)/(14000.-13300.)
                    patch[patch<0] = 0
                    patch[patch>1] = 1
                    patch = (patch*255).astype(np.uint8)
                    patch = Image.fromarray(patch).convert("RGB")

                    os.makedirs(os.path.dirname(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score))),
                                exist_ok=True) # added by gkim
                    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

        wci_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
        'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
        
        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            pass
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wci_object=wci_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=False, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
        
            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None
        
        if heatmap_args.calc_heatmap:
            compute_from_patches(wci_object=wci_object, 
                                img_transforms=img_transforms,
                                clam_pred=Y_hats[0], model=model, 
                                feature_extractor=feature_extractor, 
                                batch_size=exp_args.batch_size, **wci_kwargs, 
                                attn_save_path=save_path,  ref_scores=ref_scores)

        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue
        
        with h5py.File(save_path, 'r') as file:
            file = h5py.File(save_path, 'r')
            dset = file['attention_scores']
            coord_dset = file['coords']
            scores = dset[:]
            coords = coord_dset[:]

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
                                                                                        int(heatmap_args.blur), 
                                                                                        int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
                                                                                        float(heatmap_args.alpha), int(heatmap_args.vis_level), 
                                                                                        int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)

        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass
        
        else:           
            path_heatmap_save = os.path.join(p_slide_save_dir, heatmap_save_name)       # added by gkim because our file path has class and dataset folders                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            heatmap = drawHeatmap(scores, coords, slide_path, wci_object=wci_object,  
                                  cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
                                  binarize=heatmap_args.binarize, 
                                    blank_canvas=heatmap_args.blank_canvas,
                                    thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
                                    overlap=patch_args.overlap, 
                                    top_left=top_left, bot_right = bot_right)
            dir_heatmap_save = os.path.split(path_heatmap_save)[0] # added by gkim because our file path has class and dataset folders
            os.makedirs(dir_heatmap_save, exist_ok = True) # added by gkim because our file path has class and dataset folders 
            if heatmap_args.save_ext == 'jpg':
                heatmap.save(path_heatmap_save, quality=100)
            else:
                heatmap.save(path_heatmap_save)
        
        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wci_object.visWCI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                if heatmap_args.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)