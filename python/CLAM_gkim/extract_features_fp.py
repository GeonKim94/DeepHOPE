import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datas.dataset_h5 import Dataset_All_Bags, Whole_Colony_Bag_FP
from torch.utils.data import DataLoader
from datas.preprocess3d import TEST_AUGS_3D, TRAIN_AUGS_3D
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
from datas import colony
from models.AE3D import AE3D_SEC1


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--slide_ext', type=str, default= '.h5')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--dim_feat', type=int, default=4096)
parser.add_argument('--path_load', type=str, default=None)
args = parser.parse_args()

def compute_w_loader(file_path, output_path, wci, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	if os.path.exists(output_path):
		pass
	else:
		dataset = Whole_Colony_Bag_FP(file_path=file_path, wci=wci, pretrained=pretrained, 
			custom_downsample=custom_downsample, target_patch_size=target_patch_size, transform = TEST_AUGS_3D)
		x, y = dataset[0]
		kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
		loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
		#pdb.set_trace()

		if verbose > 0:
			print('processing {}: total of {} batches'.format(file_path,len(loader)))
		mode = 'w'
		for count, (batch, coords) in enumerate(loader):
			with torch.no_grad():	
				if count % print_every == 0:
					print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))

				batch = batch.to(device, non_blocking=True)
				if isinstance(model, nn.DataParallel):
					features = model.module.encode(batch)
				else:
					features = model.encode(batch)
				features = features.cpu().numpy()

				asset_dict = {'features': features, 'coords': coords}
				save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
				mode = 'a'
	
	return output_path


def save(net, epoch, path_save): # by Geon for custom networks
    """Save current epoch model

    Save Elements:
        model_type : arg.model
        start_epoch : current epoch
        network : network parameters
        optimizer: optimizer parameters
        best_metric : current best score

    Parameters:
        epoch : current epoch
        filename : model save file name
    """
    torch.save({"start_epoch": epoch + 1,
                "network": net.state_dict(),
                }, path_save)
    print("Model saved %d epoch" % (epoch))
    return

def remove_module_prefix(d):
    new_dict = {}
    for key, value in d.items():
        new_key = key.replace('module.', '', 1) if key.startswith('module.') else key
        new_dict[new_key] = value
    return new_dict

def load(net, path_load): # by Geon for custom networks
    """ Model load. same with save"""
    if os.path.exists(path_load) is True:
        print("Loading model from File {}".format(path_load))
        ckpoint = torch.load(path_load)
        ckpoint['network'] = remove_module_prefix(ckpoint['network'])
        net.load_state_dict(ckpoint['network'])

        return True
	
    else:
        print("Load Failed, not exists file")
        return False


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	#model = resnet50_baseline(pretrained=True)
	model = AE3D_SEC1(input_channels=1, latent_dim=args.dim_feat)
	model = model.to(device)
	
	# pdb.set_trace()
	load(model, args.path_load)
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wci = colony.colony(slide_file_path)#openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wci, 
										model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
										custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)

		pt_dir = os.path.dirname(os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
		if not os.path.exists(pt_dir):
			os.makedirs(pt_dir)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))