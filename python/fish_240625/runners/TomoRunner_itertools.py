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
import itertools

class TomoRunner(BaseRunner):
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
            self.best_metric = -1
            

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

    def train(self, train_loader, val_loader=None, test_loader=None):
        if self.arg.itertools:
            self.train_itertools(train_loader, val_loader, test_loader)

        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):

            targets = np.zeros(0, dtype=np.int8)
            correct = 0
            correct_w = 0
            self.net.train()
            if epoch == 0:
                print("the first epoch starts") # flag 1
            for i, (input_, target_, path) in enumerate(train_loader):
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, *_ = self.net(input_)
                _, idx = output_.max(dim=1)

                targets = np.append(targets, target_.cpu().numpy())
                w_acc = np.zeros(target_.shape)
                for i in range(output_.shape[1]):
                    w_acc += self.w_metric_train[i]*(target_.cpu().numpy() == i)

                #import pdb; pdb.set_trace()
                correct += np.sum(target_.cpu().numpy() == idx.cpu().numpy())
                correct_w += np.sum(np.multiply(w_acc,target_.cpu().numpy() == idx.cpu().numpy()))

                loss = self.loss(output_, target_)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if i == len(train_loader)-1:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())

            
            w_correct = np.zeros(targets.shape)
            for i in range(output_.shape[1]):
                w_correct += self.w_metric_train[i]*(targets == i)

            acc_train = correct_w / np.sum(w_correct)
            acc_train_w = correct/len(targets)

            if self.scheduler is not None:
                self.scheduler.step()
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader, acc_train = acc_train, acc_train_w = acc_train_w)
            else:
                self.save(epoch)

    def train_itertools(self, train_loader, val_loader=None, test_loader=None):

        print("\nStart Train len :", len(train_loader.dataset))

        cycle_train = itertools.cycle(train_loader)
        for epoch in range(self.start_epoch, self.epoch):
            
            count = 0
            targets = np.zeros(0, dtype=np.int8)
            correct = 0
            correct_w = 0
            self.net.train()
            if epoch == 0:
                print("the first epoch starts") # flag 1
            for (input_, target_, path) in cycle_train:
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, *_ = self.net(input_)
                _, idx = output_.max(dim=1)

                count += input_.shape[0]

                targets = np.append(targets, target_.cpu().numpy())
                w_acc = np.zeros(target_.shape)
                for i in range(output_.shape[1]):
                    w_acc += self.w_metric_train[i]*(target_.cpu().numpy() == i)

                #import pdb; pdb.set_trace()
                correct += np.sum(target_.cpu().numpy() == idx.cpu().numpy())
                correct_w += np.sum(np.multiply(w_acc,target_.cpu().numpy() == idx.cpu().numpy()))

                loss = self.loss(output_, target_)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if i == len(train_loader)-1:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())

                if count >= len(train_loader):
                    break

            
            w_correct = np.zeros(targets.shape)
            for i in range(output_.shape[1]):
                w_correct += self.w_metric_train[i]*(targets == i)

            acc_train = correct_w / np.sum(w_correct)
            acc_train_w = correct/len(targets)

            if self.scheduler is not None:
                self.scheduler.step()
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader, acc_train = acc_train, acc_train_w = acc_train_w)
            else:
                self.save(epoch)

    def _get_acc_test(self, loader, confusion=False, w_metric = (1.0,1.0)): #outputs the scores(confidence) too, will only work for batch size = 1 (GK, 200320)
        if self.arg.itertools:
            self._get_acc_test_itertools(loader, confusion, w_metric)

        correct = 0
        correct_w = 0
        preds, labels = [], []
        matdict = {}
        targets = np.zeros(0, dtype=np.int8)
        scores = np.zeros(0, dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)
        feats = np.zeros(0, dtype=np.object)

        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        if confusion:
            false_f = open(self.result_dir + "/false.txt", "w")

        for input_, target_, inputDir_ in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, feat_ = self.net(input_)

            targets = np.append(targets, target_.cpu().numpy())
            scores = np.append(scores, output_.cpu().numpy())
            paths = np.append(paths, inputDir_)
            feats = np.append(feats, feat_.cpu().numpy())

            _, idx = output_.max(dim=1)

            w_acc = np.zeros(target_.shape)
            for i in range(output_.shape[1]):
                w_acc += w_metric[i]*(target_.cpu().numpy() == i)

            correct += np.sum(np.multiply(w_acc,target_.cpu().numpy() == idx.cpu().numpy()))
            correct_w += np.sum(np.multiply(w_acc,target_.cpu().numpy() == idx.cpu().numpy()))

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
        matdict['targets'] = targets
        matdict['scores'] = scores
        matdict['paths'] = paths
        matdict['feats'] = feats
        
        if confusion:
            confusion = get_confusion(preds, labels)
            false_f.close()

        w_correct = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_correct += w_metric[i]*(targets == i)
            
        acc_w = correct_w / np.sum(w_correct)
        acc = correct/len(targets)
        return acc, acc_w, confusion, matdict

    def _get_acc_test_itertools(self, loader, confusion=False, w_metric = (1.0,1.0)): #outputs the scores(confidence) too, will only work for batch size = 1 (GK, 200320)
        cycle = itertools.cycle(loader)
        count = 0
        correct = 0
        correct_w = 0
        preds, labels = [], []
        matdict = {}
        targets = np.zeros(0, dtype=np.int8)
        scores = np.zeros(0, dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)
        feats = np.zeros(0, dtype=np.object)

        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        if confusion:
            false_f = open(self.result_dir + "/false.txt", "w")

        for (input_, target_, inputDir_) in cycle:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, feat_ = self.net(input_)

            count += 1

            targets = np.append(targets, target_.cpu().numpy())
            scores = np.append(scores, output_.cpu().numpy())
            paths = np.append(paths, inputDir_)
            feats = np.append(feats, feat_.cpu().numpy())

            _, idx = output_.max(dim=1)

            w_acc = np.zeros(target_.shape)
            for i in range(output_.shape[1]):
                w_acc += w_metric[i]*(target_.cpu().numpy() == i)

            correct += np.sum(np.multiply(w_acc,target_.cpu().numpy() == idx.cpu().numpy()))
            correct_w += np.sum(np.multiply(w_acc,target_.cpu().numpy() == idx.cpu().numpy()))

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
                        
            if count >= len(loader):
                break
            
        matdict['targets'] = targets
        matdict['scores'] = scores
        matdict['paths'] = paths
        matdict['feats'] = feats
        
        if confusion:
            confusion = get_confusion(preds, labels)
            false_f.close()

        w_correct = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_correct += w_metric[i]*(targets == i)
            
        acc_w = correct_w / np.sum(w_correct)
        acc = correct/len(targets)
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

            if acc < 0.725:
                bool_save = False
            if acc_w < 0.725:
                bool_save = False

            if train_loader is not None:
                if acc_train < 0.725:
                    bool_save = False
                if acc_train_w < 0.725:
                    bool_save = False
                
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
            
            train_acc, train_acc_w, __, matdict_train = self._get_acc_test(train_loader)
            valid_acc, valid_acc_w, __, matdict_val = self._get_acc_test(val_loader)
            test_acc, test_acc_w, test_confusion, matdict_test = self._get_acc_test(test_loader, confusion=True)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc, time=run_time)

            np.save(self.save_dir + "/test_confusion.npy", test_confusion)
            print(test_confusion)
            
            path = self.fname[0:13] + "train[%.4f]_valid[%.4f]_test[%.4f]" % (train_acc, valid_acc, test_acc)


            #path = Path(self.fname).stem
            
            if ~os.path.isdir(self.result_dir + "/" + path):
                os.mkdir(self.result_dir + "/" + path)
                np.save(self.result_dir + "/" + path+"/test_confusion.npy", test_confusion)
                io.savemat(self.result_dir + "/" + path+"/result_train.mat", matdict_train)
                io.savemat(self.result_dir + "/" + path+"/result_valid.mat", matdict_val)
                io.savemat(self.result_dir + "/" + path+"/result_test.mat", matdict_test)
            print(test_confusion)
            
        return train_acc, valid_acc, test_acc
