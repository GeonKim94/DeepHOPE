from data.dset_imgcls import dset_imgcls
from runner.runner_imgcls import runner_imgcls
from misc.arg import parse_args
from model.utils_model import get_model_imgcls
from optim.get_optim import get_optim
from optim.get_loss import get_loss
from torch.utils.data import DataLoader
from data.preprocess import *

import torch
import os

config, name_config = parse_args()
print("Configuration info:", config)

crop_z = lambda img: bottom_crop_z(img, config['data']['size_z'])
rcrop_xy = lambda img: random_crop_xy(img, (config['data']['size_xy'],config['data']['size_xy']))
ccrop_xy = lambda img: center_crop_xy(img, (config['data']['size_xy'],config['data']['size_xy']))
preps_test = [
    crop_z,
    ccrop_xy,
    clip,
    z_to_ch,
    to_tensor,
]

preps_train = [
    crop_z,
    rcrop_xy,
    flip_x,
    flip_y,
    swapaxes_xy,
    clip,
    gaussian_noise,
    z_to_ch,
    to_tensor,
]

try:
    dset_train = dset_imgcls(config['data']['dir_data']+'/data_train.yaml',preps_train)
    dset_val = dset_imgcls(config['data']['dir_data']+'/data_val.yaml',preps_test)
    dset_test = dset_imgcls(config['data']['dir_data']+'/data_test.yaml',preps_test)
except:
    dset_train = dset_imgcls(config['data']['dir_data']+'/train',preps_train)
    dset_val = dset_imgcls(config['data']['dir_data']+'/val',preps_test)
    dset_test = dset_imgcls(config['data']['dir_data']+'/test',preps_test)

get_dloader = lambda dset: DataLoader(dset, batch_size=config['optim']['batch_size'],
                                       shuffle=True, num_workers=config['compute']['cpus'],
                                       drop_last=False)

dloader_train = get_dloader(dset_train)
dloader_val = get_dloader(dset_val) 
dloader_test  = get_dloader(dset_test)

os.environ["CUDA_VISIBLE_DEVICES"] = config['compute']['gpus']
torch_device = torch.device("cuda")

net = get_model_imgcls(config['model']['type'],
                       config['model']['ch_in'],
                       config['model']['num_classes'] if config['data']['reset_class'] else len(dset_train.classnames),
                       config['data']['size_xy'],
                       config['model']['aug_arch'])
optim, sched = get_optim(net, config['optim']['optimizer_type'],
                      config['optim']['learning_rate'],
                      config['optim']['weight_decay'],
                      config['optim']['momentum'],
                      config['optim']['scheduler_type']) 

runner = runner_imgcls(net, torch_device, optim, sched,
                        get_loss(config['optim']['loss']),
                        config['model']['dir_ckpt'],
                        config['model']['fname_ckpt'],
                        config['data']['dir_infer'],
                        config['optim']['verbose'])

runner.train_repeat(dloader_train,dloader_val,dloader_test,config['optim']['epoch'])