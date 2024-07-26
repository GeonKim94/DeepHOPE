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

class TomoRunner_patchval(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger, load_fname = None, reset_loss = 0,
                 w_metric = (1.0,1.0), w_metric_test = (1.0,1.0), w_metric_train = (1.0,1.0)):
        super().__init__(arg, torch_device, logger)
        
        self.fname = load_fname
        if(self.fname == None):
            self.fname = "save_temp"#"epoch[%05d]"%(self.epoch)
        self.net = net
        self.loss = loss
        self.optim = optim
        self.arg = arg
        self.best_metric_train = -1
        self.best_metric = -1
        self.best_metric_test = -1
        self.best_metric_train_weighted = -1
        self.best_metric_weighted = -1
        self.best_metric_test_weighted = -1
        self.w_metric_train = w_metric_train
        self.w_metric = w_metric
        self.w_metric_test = w_metric_test
        self.start_time = time.time()
        
        if arg.optim == "sgd":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 16)
        else:
            self.scheduler = None
            
        self.load(load_fname)
        
        if reset_loss == 1:
            self.best_metric_train = -1
            self.best_metric = -1
            self.best_metric_test = -1
            self.best_metric_train_weighted = -1
            self.best_metric_weighted = -1
            self.best_metric_test_weighted = -1


    def save(self, epoch, filename):
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
        torch.save({"model_type": self.model_type,
                    "start_epoch": epoch + 1,
                    "network": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric,
                    "best_metric_train": self.best_metric_train,
                    "best_metric_test": self.best_metric_test,
                    "best_metric_weighted": self.best_metric_weighted,
                    "best_metric_train_weighted": self.best_metric_train_weighted,
                    "best_metric_test_weighted": self.best_metric_test_weighted
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

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
            self.optim.load_state_dict(ckpoint['optimizer'])
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

    def train(self, train_loader, val_pdset=None, test_pdset=None):
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):

            targets = np.zeros(0, dtype=np.int8)
            correct = np.zeros(0, dtype=np.int8)
            loss_sum = 0.0
            self.net.train()
            if epoch == 0:
                print("the first epoch starts") # flag 1
            for i, (input_, target_, path) in enumerate(train_loader):
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, *_ = self.net(input_)
                _, idx = output_.max(dim=1)

                targets = np.append(targets, target_.cpu().numpy())
                correct_ = idx == target_
                correct = np.append(correct, correct_.cpu().numpy())

                loss = self.loss(output_, target_)

                loss_sum += loss.item()*input_.shape[0]

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()    
                
                msg_train = "[Epoch{0:05}] training progress: {1:0.3f}%".format(epoch, (i+1)/len(train_loader)*100)
                print('\r' + msg_train, end='', flush=True)
                if i == len(train_loader)-1:
                    self.logger.log_write("train", epoch=epoch, loss=loss_sum/targets.shape[0])

            w_alldata = np.zeros(targets.shape)
            for i in range(output_.shape[1]):
                w_alldata += self.w_metric_train[i]*(targets == i)

            #import pdb; pdb.set_trace()
            acc_train_w = np.sum(np.multiply(correct, w_alldata)) / np.sum(w_alldata)
            acc_train = np.sum(correct)/len(targets)

            if self.scheduler is not None:
                self.scheduler.step()
            if val_pdset is not None:
                self.valid(epoch, val_pdset, test_pdset, acc_train = acc_train, acc_train_w = acc_train_w)
            else:
                self.save(epoch)

    def _get_acc_test(self, loader, confusion=False, w_metric = (1.0,1.0)): #outputs the scores(confidence) too, will only work for batch size = 1 (GK, 200320)

        if loader is None:  
            return None, None, None, None

        preds, labels = [], []
        matdict = {}
        targets = np.zeros(0, dtype=np.int8)
        scores = np.zeros(0, dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)
        feats = np.zeros(0, dtype=np.object)
        correct = np.zeros(0, dtype=np.int8)
        loss_sum = 0.0

        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        if confusion:
            false_f = open(self.result_dir + "/false.txt", "w")
        
        for i, (input_, target_, inputDir_) in enumerate(loader):
            #start = time.time()
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, feat_ = self.net(input_)

            targets = np.append(targets, target_.cpu().numpy())
            scores = np.append(scores, output_.cpu().numpy())
            paths = np.append(paths, inputDir_)
            feats = np.append(feats, feat_.cpu().numpy())

            _, idx = output_.max(dim=1)

            correct_ = idx == target_
            correct = np.append(correct, correct_.cpu().numpy())

            loss = self.loss(output_, target_)
            loss_sum += loss.item()*input_.shape[0]

            msg_test = "[Testing progress: {0:0.3f}%".format((i+1)/len(loader)*100)
            print('\r' + msg_test, end='', flush=True)

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()
                idx_to_class = {v: k for k, v in loader.dataset.class_to_idx.items()}
                k = 3 if len(idx_to_class) > 3 else len(idx_to_class)
                for i, (p, l) in enumerate(zip(idx.view(-1).tolist(), target_.view(-1).tolist())):
                    if p != l:
                        l = idx_to_class[l]
                        topk, indices = output_[i].topk(k)
                        indices = [(idx_to_class[i], v) for i, v in
                                   zip(indices.view(-1).tolist(), topk.view(-1).tolist())]
                        title = "Label : %s | Pred : " % (l)
                        for pred_label, pred_value in indices:
                            title += "%s : %.4f," % (pred_label, pred_value)
                        title += "\n"
                        title += "(input data dir: %s " % (inputDir_[0])
                        title += "\n"
                        false_f.write(title)         

            
            #end = time.time()
            #print("batch infered, time elapsed = {}".format(end - start))
        matdict['targets'] = targets
        matdict['scores'] = scores
        matdict['paths'] = paths
        matdict['feats'] = feats
        
        w_alldata = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_alldata += self.w_metric_train[i]*(targets == i)
        
        if confusion:
            confusion = get_confusion(preds, labels)
            false_f.close()
        else:
            confusion = None

        w_correct = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_correct += w_metric[i]*(targets == i)

        acc_w = np.sum(np.multiply(correct, w_alldata)) / np.sum(w_alldata)
        acc = np.sum(correct)/len(targets)
        return acc, acc_w, confusion, matdict


    def infer_patch(self, datasets, w_metric = (1.0,1.0)):
        print('testing...')
        dataset = datasets[0]
        matdict = {}
        targets = np.zeros(0, dtype=np.int8)
        preds_avg = np.zeros(0, dtype=np.int8)
        preds_vote = np.zeros(0, dtype=np.int8)
        scores = np.zeros((0,self.n_class), dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)

        loader_test = torch.utils.data.DataLoader(dataset, 1, num_workers = self.arg.cpus, worker_init_fn = worker_init_fn)
        input_, target_, inputDir_, coor_off_ = next(enumerate(loader_test))
        output_, feat_ = self.net(input_)
        feats = np.zeros((0, feat_.shape[1]), dtype=np.object)
        coors_off = np.zeros((0, coor_off_.shape[1]), dtype=np.int16)
        correct_avg = np.zeros(0, dtype=np.int8)
        correct_vote = np.zeros(0, dtype=np.int8)

        for dataset in self.datasets:
            loader_test = torch.utils.data.DataLoader(dataset, 1, num_workers = self.arg.cpus, worker_init_fn = worker_init_fn)
            #import pdb; pdb.set_trace()
            targets_, scores_, paths_, feats_, coors_off_, pred_avg_, pred_vote_ = self.test_once(loader_test, get_output=True)

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
        

        w_alldata = np.zeros(targets.shape)

        CACULATE acc, acc_w, confusion
        return acc, acc_w, confusion, matdict

        w_alldata = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_alldata += self.w_metric_train[i]*(targets == i)
        
        if confusion:
            confusion = get_confusion(preds, labels)
            false_f.close()
        else:
            confusion = None

        w_correct = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_correct += w_metric[i]*(targets == i)

        acc_w = np.sum(np.multiply(correct, w_alldata)) / np.sum(w_alldata)
        acc = np.sum(correct)/len(targets)
        return acc, acc_w, confusion, matdict
    


    def valid(self, epoch, val_loader, test_loader, acc_train, acc_train_w, train_loader = None):
        self.net.eval()
        with torch.no_grad():
            #import pdb; pdb.set_trace()
            if train_loader is not None:
                acc_train, acc_train_w, *_ = self._get_acc_test(train_loader, confusion = False, w_metric = self.w_metric_train)
            acc, acc_w,*_ = self._get_acc_test(val_loader, confusion = False, w_metric = self.w_metric)
            acc_test, acc_test_w, *_ = self._get_acc_test(test_loader, confusion = False, w_metric = self.w_metric_test)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=acc_test)
            
            bool_save = False
            
            if train_loader is not None:
                if acc_train >= self.best_metric_train:
                    if epoch % 25 == 0:
                        self.best_metric_train = acc_train
                        bool_save = True
            
            if acc >= self.best_metric:
                self.best_metric = acc
                bool_save = True
            
            if acc_test >= self.best_metric_test:
                self.best_metric_test = acc_test
                bool_save = True

            if train_loader is not None:
                if acc+acc_train >= self.best_metric+self.best_metric_train:
                    bool_save = True
            
            if train_loader is not None:
                if acc_train_w >= self.best_metric_train_weighted:
                    if epoch % 25 == 0:
                        self.best_metric_train_weighted = acc_train_w
                        bool_save = True

            if acc_w >= self.best_metric_weighted:
                self.best_metric_weighted = acc_w
                bool_save = True

            if acc_test_w >= self.best_metric_test_weighted:
                self.best_metric_test_weighted = acc_test_w
                bool_save = True
                    
            if train_loader is not None:
                if acc_w+acc_train_w >= self.best_metric_weighted+self.best_metric_train_weighted:
                    bool_save = True

            if acc > self.best_metric - 0.0125:
                bool_save = True
            if acc_w > self.best_metric_weighted - 0.0125:
                bool_save = True

            #if acc < 0.675:
            #    bool_save = False
            #if acc_w < 0.675:
            #    bool_save = False

            #if train_loader is not None:
            #    if acc_train < 0.725:
            #        bool_save = False
            #    if acc_train_w < 0.725:
            #        bool_save = False
                
            if bool_save:
                self.save(epoch, "epoch[%05d]_tr[%.3f]_va[%.3f]_te[%.3f]_trW[%.3f]_vaW[%.3f]_teW[%.3f]"
                    % (epoch, acc_train, acc, acc_test, acc_train_w, acc_w, acc_test_w))
                
            if epoch%25 == 0:
                print('best validation acc: {:.4f}\nbest test acc: {:.4f}'.format(self.best_metric, self.best_metric_test))
                
    def test(self, train_loader, val_loader, test_loader):
        
        print("\n Start Test")
        #self.load()
        self.net.eval()
        with torch.no_grad():
            
            train_acc, train_acc_w, __, matdict_train = self._get_acc_test(train_loader, confusion = False, w_metric = self.w_metric_train)
            valid_acc, valid_acc_w, __, matdict_val = self._get_acc_test(val_loader, confusion = False, w_metric = self.w_metric)
            test_acc, test_acc_w, __, matdict_test = self._get_acc_test(test_loader, confusion=False, w_metric = self.w_metric_test)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc, time=run_time)

            #np.save(self.save_dir + "/test_confusion.npy", test_confusion)
            #print(test_confusion)

            if train_acc is None:
                train_acc = 0
            if valid_acc is None:
                valid_acc = 0
            
            path = self.fname[0:13] + "train[%.4f]_valid[%.4f]_test[%.4f]" % (train_acc, valid_acc, test_acc)


            #path = Path(self.fname).stem
            
            if ~os.path.isdir(self.result_dir + "/" + path):
                os.mkdir(self.result_dir + "/" + path)
                #np.save(self.result_dir + "/" + path+"/test_confusion.npy", test_confusion)
                if matdict_train is not None:
                    io.savemat(self.result_dir + "/" + path+"/result_train.mat", matdict_train)
                if matdict_val is not None:
                    io.savemat(self.result_dir + "/" + path+"/result_valid.mat", matdict_val)
                io.savemat(self.result_dir + "/" + path+"/result_test.mat", matdict_test)
            
        return train_acc, valid_acc, test_acc


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) // worker_info.num_workers
    
    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]