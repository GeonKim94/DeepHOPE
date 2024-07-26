import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from .BaseRunner import BaseRunner
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import time
from utils import get_confusion
from pathlib import Path
import scipy.io as io

class TomoRunner_patch(BaseRunner):
    def __init__(self, arg, net, num_workers, datasets_test, torch_device, loss, load_fname = None,
            w_metric = (1.0,1.0), w_metric_test = (1.0,1.0), w_metric_train = (1.0,1.0), logger = None, n_class = 2):
        
        arg.epoch = 0
        super().__init__(arg, torch_device, logger)

        self.num_workers = num_workers
        self.net = net
        self.torch_device = torch_device
        self.loss = loss
        self.w_metric = w_metric
        self.w_metric_test = w_metric_test
        self.w_metric_train = w_metric_train
        self.datasets_test, self.classes, self.class_to_idx = datasets_test
        # value_list = [value for value in self.class_to_idx.values()]
        # value_list = set(value_list)
        #self.n_class = len(value_list)
        self.n_class = n_class
        self.fname = load_fname

        self.load(load_fname)

    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            print(len(filenames))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[self.arg.testfile])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

            self.net.load_state_dict(ckpoint['network'])
            #self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f" % (
            ckpoint["model_type"], self.start_epoch, self.best_metric))
   
            try:
                self.best_metric_train = ckpoint["best_metric_train"]
            except:
                pass
            try:
                self.best_metric_test = ckpoint["best_metric_test"]
            except:
                pass
            try:
                self.best_metric_weighted = ckpoint["best_metric_weighted"]
            except:
                pass
            try:
                self.best_metric_train_weighted = ckpoint["best_metric_train_weighted"]
            except:
                pass
            try:
                self.best_metric_test_weighted = ckpoint["best_metric_test_weighted"]
            except:
                pass
        else:
            print("Load Failed, not exists file")

    def test_once(self, loader, w_metric = (1.0,1.0)): #outputs the scores(confidence) too, will only work for batch size = 1 (GK, 200320)

        if loader is None:  
            return None, None, None, None, None
        targets = np.zeros(0, dtype=np.int8)
        scores = np.zeros((0,self.n_class), dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)

        input_, target_, inputDir_, coor_off_ = next(iter(loader))
        output_, feat_ = self.net(input_)
        feats = np.zeros((0, feat_.shape[1]), dtype=np.object)
        coors_off = np.zeros((0, len(coor_off_)), dtype=np.int16)
        correct = np.zeros(0, dtype=np.int8)
        loss_sum = 0.0
        count_batch = 0

        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
            
        with torch.no_grad():
            self.net.eval()
            for input_, target_, inputDir_, coor_off_ in loader:#i, (input_, target_, inputDir_, coor_off_) in enumerate(loader):
                #start = time.time()
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, feat_ = self.net(input_)

                if count_batch == 0:
                    targets = np.append(targets, target_.cpu().numpy())
                scores = np.append(scores, output_.cpu().numpy(), axis = 0)
                paths = np.append(paths, inputDir_)
                feats = np.append(feats, np.expand_dims(feat_.cpu().numpy(),0))
                coors_off = np.append(coors_off, np.asarray(coor_off_))

                _, idx = output_.max(dim=1)

                correct_ = idx == target_
                correct = np.append(correct, correct_.cpu().numpy())

                loss = self.loss(output_, target_)
                loss_sum += loss.item()*input_.shape[0]

                msg_test = "[Testing progress: {0:0.3f}%".format((count_batch+1)/len(loader)*100)
                print('\r' + msg_test, end='', flush=True)
                count_batch += 1

        scores_avg = np.average(scores, 0)
        max_score = np.max(scores_avg, axis = 0)
        pred_avg = np.where(scores_avg == max_score)[0]
        #import pdb;pdb.set_trace()
        preds_all = np.argmax(scores, axis = 1)
        class_preds, counts = np.unique(preds_all, return_counts=True)
        ind = np.argmax(counts)
        pred_vote = class_preds[ind]
        if pred_vote.size>1:
            pred_vote = pred_vote(np.argmax(scores_avg[0][pred_vote], axis = 1))
           
        return targets, scores, paths, feats, coors_off, pred_avg, pred_vote


    def test(self):
        print('testing...')
        dataset = self.datasets_test[0]
        matdict = {}
        targets = np.zeros(0, dtype=np.int8)
        preds_avg = np.zeros(0, dtype=np.int8)
        preds_vote = np.zeros(0, dtype=np.int8)
        scores = np.zeros((0,self.n_class), dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)

        # import pdb; pdb.set_trace()
        loader_test = torch.utils.data.DataLoader(dataset, 1, num_workers = self.num_workers, worker_init_fn = worker_init_fn)
        #for input_, target_, inputDir_, coor_off_ in enumerate(loader_test):#enumerate(loader_test):
        #    break
        input_, target_, inputDir_, coor_off_ = next(iter(loader_test))
        output_, feat_ = self.net(input_)
        feats = np.zeros((0, feat_.shape[1]), dtype=np.object)
        coors_off = np.zeros((0, len(coor_off_)), dtype=np.int16)
        correct_avg = np.zeros(0, dtype=np.int8)
        correct_vote = np.zeros(0, dtype=np.int8)

        for dataset in self.datasets_test:
            loader_test = torch.utils.data.DataLoader(dataset, 1, num_workers = self.num_workers, worker_init_fn = worker_init_fn)
            # import pdb; pdb.set_trace()
            targets_, scores_, paths_, feats_, coors_off_, pred_avg_, pred_vote_ = self.test_once(loader_test)

            targets = np.append(targets, targets_)
            scores = np.append(scores, scores_)
            paths = np.append(paths, paths_)
            feats = np.append(feats, feats_)
            coors_off = np.append(coors_off, coors_off_)
            correct_avg = np.append(correct_avg, pred_avg_ == targets_)
            correct_vote = np.append(correct_vote, pred_vote_ == targets_)


        matdict['targets'] = targets
        matdict['scores'] = scores
        matdict['paths'] = paths
        matdict['feats'] = feats
        matdict['coors_off'] = coors_off
        matdict['preds_avg'] = preds_avg
        matdict['preds_vote'] = preds_vote

        path = self.fname[0:13] + "temp"#train[%.4f]_valid[%.4f]_test[%.4f]" % (train_acc, valid_acc, test_acc)
        if ~os.path.isdir(self.result_dir + "/" + path):
            os.mkdir(self.result_dir + "/" + path)
        #np.save(self.result_dir + "/" + path+"/test_confusion.npy", test_confusion)
        if matdict is not None:
            io.savemat(self.result_dir + "/" + path+"/result_patch.mat", matdict)
            


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
    
    if window_size[2] == 1:
        windz = np.ones(window_size)
    else:
        windz = _spline_window(window_size[2], power)
        windz = np.expand_dims(np.expand_dims(windz, 0), 0)
    #print(windz)
    return windx*windy*windz