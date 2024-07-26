import h5py
import numpy as np
import os
import pdb
from PIL import Image
import math
import cv2

def initialize_hdf5_bag(first_patch, save_coord=False):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(first_patch.values())
    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis,...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs', 
                                shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x,y)

    file.close()
    return file_path

def savePatchIter_bag_hdf5(patch):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path= tuple(patch.values())
    img_patch = np.array(img_patch)[np.newaxis,...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "a")

    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    if 'coords' in file:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
        coord_dset[-img_shape[0]:] = (x,y)

    file.close()

def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False
    
def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file = h5py.File(output_path, mode)
    #pdb.set_trace()
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

# GKim's note: Below are edited since 20240429 for heatmap visulaization

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end_value = np.quantile(scores, end)
        start_value = np.quantile(scores, start)
    else:
        end_value = end
        start_value = start
    score_window = np.logical_and(scores >= start_value, scores <= end_value)
    indices = np.where(score_window)[0]
    if len(indices) < 1:
        return -1 
    else:
        return np.random.choice(indices, min(k, len(indices)), replace=False)


def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset