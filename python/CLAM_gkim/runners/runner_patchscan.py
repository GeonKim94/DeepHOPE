import torch
import math
import os
# import utils.set_requires_grad as set_requires_grad
from torchvision import transforms
# from utils.gettime import gettime
import itertools
import scipy.io as io
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

class runner_supervised_patch(object):
    def __init__(self, net, loss, w_loss, datasets_test, num_workers, logger, device, path_ckpt, path_log, fname_load):
        self.num_workers = num_workers
        self.net = net
        self.datasets_test = datasets_test
        #self.enum_test = itertools.cycle(self.loader_test)# enumerate(self.loader_train)#
        self.logger = logger
        self.path_log = path_log
        self.path_ckpt = path_ckpt
        self.device = device
        self.epoch = 0
        self.loss = loss
        self.w_loss = w_loss
        self.loss_best_train = math.inf
        self.loss_best_val = math.inf
        self.loss_best_test = math.inf

        self.load_dict(fname_load)

    def save_dict(self, fname_dict):
        if not os.path.exists(self.path_ckpt):
            os.mkdir(self.path_ckpt)
        torch.save({'net': self.net.state_dict(),
                    'optim': self.optim.state_dict(),
                    'loss_best_train': self.loss_best_train,
                    'loss_best_val': self.loss_best_val,
                    'loss_best_test': self.loss_best_test,
                    'epoch': self.epoch},
                    os.path.join(self.path_ckpt, fname_dict))

    def load_dict(self, fname_ckpt):
        if os.path.exists(os.path.join(self.path_ckpt, fname_ckpt)):
            dict_ckpt = torch.load(os.path.join(self.path_ckpt, fname_ckpt))
            try:
                self.net.load_state_dict(dict_ckpt['net'])
            except:
                print("[WARNING] parameters for network not matching -> failed to load")
            
            # try:
            #     self.optim.load_state_dict(dict_ckpt['optim'])
            # except:
            #     print("[WARNING] parameters for optimizer not matching -> failed to load")
            
            try:
                self.epoch = dict_ckpt['epoch']
            except:
                print("[WARNING] parameters for epoch not matching -> failed to load")

            try:
                self.loss_best_train = dict_ckpt['loss_best_train']
                self.loss_best_val = dict_ckpt['loss_best_val']
                self.loss_best_test = dict_ckpt['loss_best_test']
            except:
                print("[WARNING] parameters for loss not matching -> failed to load")
        else:
            print('[WARNING] loading the final model: file %s does not exists in %s'%(fname_ckpt, self.path_ckpt))
            list_ckpt = os.listdir(self.path_ckpt)
            list_ckpt = [f for f in list_ckpt if f.endswith('pth')]
            list_ckpt = [os.path.join(self.path_ckpt,f) for f in list_ckpt]
            if len(list_ckpt) > 0:
                list_ckpt.sort(key=lambda x: os.path.getmtime(x))
                self.load_dict(list_ckpt[-1])
                print('Model loaded: %s'%list_ckpt[-1])
            else:
                print('[WARNING] no ckpt file found: training from scrap')
    
        if os.path.exists(self.path_ckpt+"history.mat"):
            data = io.loadmat(self.path_ckpt+"history.mat")
            self.epochs = data['epochs'].tolist()
            self.losses_train = data['losses_train'].tolist()
            self.losses_val = data['losses_val'].tolist()
            self.losses_test = data['losses_test'].tolist()
    
    def test_once(self, loader, get_output = True, writer = None, freq_write = 1):
        print('testing once...')
        # old version enumerates dataloader every epoch
        loss_sum = 0.0
        count_sum = 0.0
        size_stitch = loader.dataset.input.shape
        size_patch = loader.dataset.patch_size
        output_stitch = np.swapaxes(np.zeros((size_stitch)),0,2).astype(np.float32)
        map_stitch = np.swapaxes(np.zeros((size_stitch)),0,2).astype(np.float32)
        window_stitch = (np.swapaxes(_window_3d(size_patch,2),0,2)+1.0e-011).astype(np.float32)
        with torch.no_grad():
            self.net.eval()
            for idx_batch, (input_, label_, path_, coor_off_) in enumerate(loader):
                print('idx_batch: %03d'%idx_batch)
                input_, label_ = input_.to(self.device), label_.to(self.device)
                count = input_.shape[0]
                output_ = self.net(input_)
                #import pdb; pdb.set_trace()
                output_stitch[coor_off_[2]:coor_off_[2]+size_patch[2],
                              coor_off_[1]:coor_off_[1]+size_patch[1],
                              coor_off_[0]:coor_off_[0]+size_patch[0]] += np.multiply(output_[0].cpu().numpy(),window_stitch).astype(np.float32) # only batch size 1 is possible here

                map_stitch[coor_off_[2]:coor_off_[2]+size_patch[2],
                              coor_off_[1]:coor_off_[1]+size_patch[1],
                              coor_off_[0]:coor_off_[0]+size_patch[0]] += window_stitch#np.ones(output_[0].cpu().numpy().shape)

        if get_output == True:
            dict_save = {'output': np.divide(output_stitch,map_stitch),
                            'path': path_[0]}
            dir_save = os.path.join(self.path_log, 'results_epoch%04d' %self.epoch)
            if not os.path.isdir(dir_save):
                os.makedirs(dir_save)
            dir_save = os.path.join(dir_save, os.path.basename(os.path.dirname(path_[0]))) # add which path
            if not os.path.isdir(dir_save):
                os.makedirs(dir_save)
            io.savemat(os.path.join(dir_save,os.path.basename(path_[0]).replace('.h5','.mat')), dict_save)

    def test(self):
        print('testing...')
        for dataset in self.datasets_test:
            loader_test = torch.utils.data.DataLoader(dataset, 1, num_workers = self.num_workers, worker_init_fn = worker_init_fn)
            #import pdb; pdb.set_trace()
            self.test_once(loader_test, get_output=True)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers
    
    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]


def _spline_window(window_size, power=2):
    intersection = window_size // 4
    tri = 1. - np.abs((window_size - 1) / 2. - np.arange(0, window_size)) / ((window_size - 1) / 2.)
    wind_outer = np.power(tri * 2, power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - np.abs(2 * (tri - 1)) ** power / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

def _window_2d(window_size=128, power=2):
    """
        Make a 1D window function, then infer and return a 2D window function.
        Done with an augmentation, and self multiplication with its transpose.
        Could be generalized to more dimensions.
        """
    wind = _spline_window(window_size, power)
    wind = np.expand_dims(np.expand_dims(wind, 1), 2)
    wind = wind * wind.transpose(1, 0, 2)
    return wind.transpose(2, 0, 1)


def _window_3d(window_size=(512,512,32), power=2):
    """
        Make a 1D window function, then infer and return a 2D window function.
        Done with an augmentation, and self multiplication with its transpose.
        Could be generalized to more dimensions.
        """
    
    windx = _spline_window(window_size[0], power)
    windx = np.expand_dims(np.expand_dims(windx, 1), 2)

    windy = _spline_window(window_size[1], power)
    windy = np.expand_dims(np.expand_dims(windy, 0), 2)
    
    windz = _spline_window(window_size[2], power)
    windz = np.expand_dims(np.expand_dims(windz, 0), 0)

    # wind = _spline_window(window_size, power)
    # wind = np.expand_dims(np.expand_dims(wind, 1), 2)
    # wind = wind * wind.transpose(1, 0, 2)
    return windx*windy*windz