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

class LymphoRunner(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger, load_fname = None, reset_loss = 0, w_metric = (1.0,1.0), w_metric_test = (1.0,1.0)):
        super().__init__(arg, torch_device, logger)
        
        self.fname = load_fname
        if(self.fname == None):
            self.fname = "save_temp"#"epoch[%05d]"%(self.epoch)
        self.net = net
        self.loss = loss
        self.optim = optim
        self.arg = arg
        self.best_metric = -1
        self.start_time = time.time()
        self.w_metric = w_metric
        self.w_metric_test = w_metric_test
        
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
                    "best_metric": self.best_metric
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
        else:
            print("Load Failed, not exists file")

    def train(self, train_loader, val_loader=None, test_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):
            self.net.train()
            if epoch == 0:
                print("the first epoch starts") # flag 1
            for i, (input_, target_, path) in enumerate(train_loader):
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, *_ = self.net(input_)
                loss = self.loss(output_, target_)


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


                if i == len(train_loader)-1:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())
            if self.scheduler is not None:
                self.scheduler.step()
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader)
            else:
                self.save(epoch)

    def _get_acc_test(self, loader, confusion=False, w_metric = (1.0,1.0)): #outputs the scores(confidence) too, will only work for batch size = 1 (GK, 200320)
        correct = 0
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
        return correct / np.sum(w_correct), confusion, matdict

    def _get_acc(self, loader, confusion=False): # 220428 this function somehow yields different accuracy from _get_acc_test (cannot understand)
        correct = 0
        preds, labels = [], []
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        if confusion:
            false_f = open(self.result_dir + "/false.txt", "w")

        for input_, target_, inputDir_ in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, *_ = self.net(input_)

            _, idx = output_.max(dim=1)
            correct += torch.sum(target_ == idx).float().cpu().item()

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

        if confusion:
            confusion = get_confusion(preds, labels)
            false_f.close()

        return correct / len(loader.dataset), confusion



    def valid(self, epoch, val_loader, test_loader):
        self.net.eval()
        with torch.no_grad():
            acc, *_ = self._get_acc_test(val_loader, confusion = False, w_metric = self.w_metric)
            test_acc, *_ = self._get_acc_test(test_loader, confusion = False)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc)
            if acc > self.best_metric and acc > 0.50:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, acc, test_acc))
            elif acc > self.best_metric - 0.01 and acc > 0.50:
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, acc, test_acc))
            elif test_acc > 0.9:
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, acc, test_acc))
            elif epoch % 10 == 0:
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, acc, test_acc))
                


    def _get_acc_25d(self, loader):
        patch_correct_sum = 0
        preds, labels = [], []

        cell_target = {}
        cell_correct = defaultdict(lambda: 0)
        for input_, target_, path in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, *_ = self.net(input_)

            _, idx = output_.max(dim=1)
            patch_correct = torch.sum(target_ == idx).float().cpu().item()
            patch_correct_sum += patch_correct

            for b in range(len(path)):
                cell_correct[path[b]] += output_[b]
                cell_target[path[b]] = target_[b]

        correct = 0
        for k, v in cell_correct.items():
            target_ = cell_target[k]
            _, idx = v.max(dim=0)
            correct += (target_ == idx).float().cpu().item()

            preds += idx.view(-1).tolist()
            labels += target_.view(-1).tolist()

        acc = correct / len(cell_correct.keys())

        idx_to_cls = {v: k for k, v in loader.dataset.class_to_idx.items()}
        preds = [idx_to_cls[i] for i in preds]
        labels = [idx_to_cls[i] for i in labels]
        a = confusion_matrix(labels, preds, labels=loader.dataset.classes)
        return acc, patch_correct_sum / len(loader.dataset), a

    def _test_25d(self, train_loader, val_loader, test_loader):
        print("\n Start Test")
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc, train_patch, _ = self._get_acc_25d(train_loader)
            valid_acc, valid_patch, _ = self._get_acc_25d(val_loader)
            test_acc, test_patch, test_confusion = self._get_acc_25d(test_loader)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)

            np.save(self.save_dir + "/test_confusion.npy", test_confusion)
            print(test_confusion)

    def test(self, train_loader, val_loader, test_loader):
        

        if self.arg.dim == "25d":
            return self._test_25d(train_loader, val_loader, test_loader)

        print("\n Start Test")
        #self.load()
        self.net.eval()
        with torch.no_grad():
            valid_acc, _, matdict_val = self._get_acc_test(val_loader)
            test_acc, test_confusion, matdict_test = self._get_acc_test(test_loader, confusion=True)
            train_acc, _, matdict_train = self._get_acc_test(train_loader)

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
