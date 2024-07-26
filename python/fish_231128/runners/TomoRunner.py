import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from .BaseRunner import BaseRunner
from collections import defaultdict
import time
#from utils import get_confusion
from pathlib import Path
import scipy.io as io
import math

class TomoRunner(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger, load_fname = None, reset_loss = 0,
                 w_metric = (1.0,1.0), w_metric_test = (1.0,1.0), w_metric_train = (1.0,1.0), n_class = 2):
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
        self.n_class = n_class

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

            first_key = next(iter(ckpoint['network']))  # Get the first key of the dictionary
            if isinstance(self.net, nn.DataParallel):
                if first_key.startswith("module."):
                    self.net.load_state_dict(ckpoint['network'])
                else:
                    new_dict = {"module." + key: value for key, value in ckpoint['network'].items()}
                    self.net.load_state_dict(new_dict)
            else:
                if first_key.startswith("module."):
                    new_dict = {key.replace("module.", ""): value for key, value in ckpoint['network'].items()}
                    self.net.load_state_dict(new_dict)
                else:
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
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader, acc_train = acc_train, acc_train_w = acc_train_w)
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

#         if confusion:
#             false_f = open(self.result_dir + "/false.txt", "w")
        
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

#             if confusion:
#                 preds += idx.view(-1).tolist()
#                 labels += target_.view(-1).tolist()
#                 idx_to_class = {v: k for k, v in loader.dataset.class_to_idx.items()}
#                 k = 3 if len(idx_to_class) > 3 else len(idx_to_class)
#                 for i, (p, l) in enumerate(zip(idx.view(-1).tolist(), target_.view(-1).tolist())):
#                     if p != l:
#                         l = idx_to_class[l]
#                         topk, indices = output_[i].topk(k)
#                         indices = [(idx_to_class[i], v) for i, v in
#                                    zip(indices.view(-1).tolist(), topk.view(-1).tolist())]
#                         title = "Label : %s | Pred : " % (l)
#                         for pred_label, pred_value in indices:
#                             title += "%s : %.4f," % (pred_label, pred_value)
#                         title += "\n"
#                         title += "(input data dir: %s " % (inputDir_[0])
#                         title += "\n"
#                         false_f.write(title)         

            
            #end = time.time()
            #print("batch infered, time elapsed = {}".format(end - start))
        matdict['targets'] = targets
        matdict['scores'] = scores
        matdict['paths'] = paths
        matdict['feats'] = feats
        
        w_alldata = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_alldata += self.w_metric_train[i]*(targets == i)
        
#         if confusion:
#             confusion = get_confusion(preds, labels)
#             false_f.close()
#         else:
        confusion = None

        w_correct = np.zeros(targets.shape)
        for i in range(output_.shape[1]):
            w_correct += w_metric[i]*(targets == i)

        acc_w = np.sum(np.multiply(correct, w_alldata)) / np.sum(w_alldata)
        acc = np.sum(correct)/len(targets)
        return acc, acc_w, confusion, matdict

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
            if test_acc is None:
                test_acc = 0
            if train_acc_w is None:
                train_acc_w = 0
            if valid_acc_w is None:
                valid_acc_w = 0
            if test_acc_w is None:
                test_acc_w = 0
            path = self.fname[0:13] + "tr[%.4f]_va[%.4f]_te[%.4f]_trW[%.4f]_vaW[%.4f]_teW[%.4f]" % (train_acc, valid_acc, test_acc,train_acc_w, valid_acc_w, test_acc_w)
            # path = self.fname[0:13] + "train[%.4f]_valid[%.4f]_test[%.4f]" % (train_acc, valid_acc, test_acc)


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
    

#######################################################################

    def train_patchval(self, train_loader, val_pdsets = None, test_pdsets=None):
        print("\nStart Train (patchval) len :", len(train_loader.dataset))
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
            if val_pdsets is not None:
                self.valid_patch(epoch, test_pdsets, test_pdsets, acc_train = acc_train, acc_train_w = acc_train_w)
            else:
                self.save(epoch)

    

    def _get_acc_onepatch(self, loader):
        if loader is None:  
            return None, None, None, None, None
        targets = np.zeros(0, dtype=np.int8)
        scores = np.zeros((0,self.n_class), dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)
        losses = np.zeros(0, dtype=np.object)
        
        input_, target_, inputDir_, coor_off_ = next(loader)
        output_, feat_ = self.net(input_)
        feats = np.zeros((0, feat_.shape[1]), dtype=np.object)
        coors_off = np.zeros((0, coor_off_.shape[1]), dtype=np.int16)
        correct = np.zeros(0, dtype=np.int8)
        count_batch = 0
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
            
        with torch.no_grad():
            self.net.eval()
            for input_, target_, inputDir_, coor_off_ in loader:
                #start = time.time()
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, feat_ = self.net(input_)

                if count_batch == 0:
                    targets = np.append(targets, target_.cpu().numpy())
                scores = np.append(scores, np.expand_dims(output_.cpu().numpy(),0))
                paths = np.append(paths, inputDir_)
                feats = np.append(feats, np.expand_dims(feat_.cpu().numpy(),0))
                coors_off = np.append(coors_off, np.asarray(coor_off_))

                _, idx = output_.max(dim=1)

                correct_ = idx == target_
                correct = np.append(correct, correct_.cpu().numpy())

                loss = self.loss(output_, target_)
                losses = np.append(losses, loss.cpu().numpy())
                # loss_sum += loss.item()*input_.shape[0]

                msg_test = "[Testing progress: {0:0.3f}%".format((i+1)/len(loader)*100)
                print('\r' + msg_test, end='', flush=True)
                count_batch += 1

        scores_avg = np.average(scores, 0)
        max_score = np.max(scores_avg, axis = 1)
        pred_avg = np.where(scores_avg == max_score)[1]

        preds_all = np.argmax(scores, axis = 1)
        class_preds, counts = np.unique(preds_all, return_counts=True)
        ind = np.argmax(counts)
        pred_vote = class_preds[ind]
        if len(pred_vote)>1:
            pred_vote = pred_vote(np.argmax(scores_avg[0][pred_vote], axis = 1))
           
        return targets, scores, paths, feats, coors_off, pred_avg, pred_vote, losses


    def _get_acc_patches(self, datasets):
        if datasets is None:  
            return None, None, None, None
        matdict = {}
        dataset = datasets[0]
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
        loss_sum = 0.0
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        
        for dataset in datasets:
            loader = torch.utils.data.DataLoader(dataset, 1, num_workers = self.num_workers, worker_init_fn = worker_init_fn)
            #import pdb; pdb.set_trace()
            targets_, scores_, paths_, feats_, coors_off_, pred_avg_, pred_vote_, losses_ = self._get_acc_onepatch(loader)

            targets = np.append(targets, targets_)
            scores = np.append(scores, scores_)
            paths = np.append(paths, paths_)
            feats = np.append(feats, feats_)
            coors_off = np.append(coors_off, coors_off_)
            correct_avg = np.append(correct_avg, pred_avg_ == targets_)
            correct_vote = np.append(correct_vote, pred_vote_ == targets_)
            losses = np.append(losses, losses_)


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

    def test_patch(self, val_pdsets, test_pdsets, train_pdsets = None):
        print('testing...')
        dataset = val_pdsets[0]

        loader_test = torch.utils.data.DataLoader(dataset, 1, num_workers = self.arg.cpus, worker_init_fn = worker_init_fn)
        input_, target_, inputDir_, coor_off_ = next(enumerate(loader_test))
        output_, feat_ = self.net(input_)
        feat_shape = feat_.shape[1]
        score_shape = output_.shape[1]

        coor_shape = coor_off_.shape[1]

        self.net.eval()
        with torch.no_grad():
            targets, scores, paths, feats, coors_off, correct_avg, correct_vote, acc_avg, acc_w_avg, acc_vote, acc_w_vote = self.eval_pdsets(val_pdsets, feat_shape, coor_shape, score_shape)
            targets_test, scores_test, paths_test, feats_test, coors_off_test, correct_avg_test, correct_vote_test, acc_test_avg, acc_test_w_avg, acc_test_vote, acc_test_w_vote = self.eval_pdsets(test_pdsets, feat_shape, coor_shape, score_shape)
            if train_pdsets is not None:
                targets_train, scores_train, paths_train, feats_train, coors_off_train, correct_avg_train, correct_vote_train, acc_train_avg, acc_train_w_avg, acc_train_vote, acc_train_w_vote = self.eval_pdsets(test_pdsets, feat_shape, coor_shape, score_shape)
    
        matdict = {}
        matdict['targets'] = targets
        matdict['scores'] = scores
        matdict['paths'] = paths
        matdict['feats'] = feats
        matdict['coors_off'] = coors_off
        matdict['correct_avg'] = correct_avg
        matdict['correct_vote'] = correct_vote

        matdict_test = {}
        matdict_test['targets'] = targets_test
        matdict_test['scores'] = scores_test
        matdict_test['paths'] = paths_test
        matdict_test['feats'] = feats_test
        matdict_test['coors_off'] = coors_off_test
        matdict_test['correct_avg'] = correct_avg_test
        matdict_test['correct_vote'] = correct_vote_test

        if train_pdsets is not None:
            matdict_train = {}
            matdict_train['targets'] = targets_train
            matdict_train['scores'] = scores_train
            matdict_train['paths'] = paths_train
            matdict_train['feats'] = feats_train
            matdict_train['coors_off'] = coors_off_train
            matdict_train['correct_avg'] = correct_avg_train
            matdict_train['correct_vote'] = correct_vote_train
        else:
            acc_train_avg = 0
            acc_train_vote = 0

        path = self.fname[0:13] + "train[%.4f]_valid[%.4f]_test[%.4f]" % (acc_train_avg, acc_avg, acc_test_avg)

        if ~os.path.isdir(self.result_dir + "/" + path):
            os.mkdir(self.result_dir + "/" + path)
            #np.save(self.result_dir + "/" + path+"/test_confusion.npy", test_confusion)
            if matdict_train is not None:
                io.savemat(self.result_dir + "/" + path+"/result_train.mat", matdict_train)
            if matdict is not None:
                io.savemat(self.result_dir + "/" + path+"/result_valid.mat", matdict)
            io.savemat(self.result_dir + "/" + path+"/result_test.mat", matdict_test)
            
        return acc_train_avg, acc_avg, acc_test_avg, acc_train_vote, acc_vote, acc_test_vote



    def eval_pdsets(self, pdsets, feat_shape, coor_shape, score_shape):
        self.net.eval()
        with torch.no_grad():
            targets = np.zeros(0, dtype=np.int8)
            preds_avg = np.zeros(0, dtype=np.int8)
            preds_vote = np.zeros(0, dtype=np.int8)
            scores = np.zeros((0,self.n_class), dtype=np.float32)
            paths = np.zeros(0, dtype=np.object)
            feats = np.zeros((0, feat_shape), dtype=np.object)
            coors_off = np.zeros((0, coor_shape), dtype=np.int16)
            correct_avg = np.zeros(0, dtype=np.int8)
            correct_vote = np.zeros(0, dtype=np.int8)
            for dataset in pdsets:
                loader = torch.utils.data.DataLoader(dataset, 1, num_workers = self.arg.cpus, worker_init_fn = worker_init_fn)
                #targets_, _, _, _, coors_off_, pred_avg_, pred_vote_ = self.eval_patches(loader)
                targets_, scores_, paths_, feats_, coors_off_, pred_avg_, pred_vote_ = self.eval_patches(loader)

                targets = np.append(targets, targets_)
                scores = np.append(scores, scores_)
                paths = np.append(paths, paths_)
                feats = np.append(feats, feats_)
                coors_off = np.append(coors_off, coors_off_)
                correct_avg = np.append(correct_avg, pred_avg_ == targets_)
                correct_vote = np.append(correct_vote, pred_vote_ == targets_)
            w_correct = np.zeros(targets.shape)
            for i in range(score_shape):
                w_correct += self.w_metric[i]*(targets == i)
            acc_w_avg = np.sum(np.multiply(correct_avg, w_correct)) / np.sum(w_correct)
            acc_avg = np.sum(correct_vote)/targets.size
            acc_w_vote = np.sum(np.multiply(correct_vote, w_correct)) / np.sum(w_correct)
            acc_vote = np.sum(correct_vote)/targets.size
        return targets, scores, paths, feats, coors_off, correct_avg, correct_vote, acc_avg, acc_w_avg, acc_vote, acc_w_vote
        



    def eval_patches(self, loader):
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


    def valid_patch(self, epoch, val_pdsets, test_pdsets, acc_train, acc_train_w, train_loader = None):
        print('testing...')
        dataset = val_pdsets[0]
        matdict = {}

        loader_test = torch.utils.data.DataLoader(dataset, 1, num_workers = self.arg.cpus, worker_init_fn = worker_init_fn)
        input_, target_, inputDir_, coor_off_ = next(enumerate(loader_test))
        output_, feat_ = self.net(input_)
        score_shape = output_.shape[1]
        feat_shape = feat_.shape[1]

        coor_shape = coor_off_.shape[1]

        self.net.eval()
        with torch.no_grad():
            # targets, scores, paths, feats, coors_off, correct_avg, correct_vote, acc_avg, acc_w_avg, acc_vote, acc_w_vote = self.eval_pdsets(val_pdsets, feat_shape, coor_shape, score_shape)
            # targets_test, scores_test, paths_test, feats_test, coors_off_test, correct_avg_test, correct_vote_test, acc_test_avg, acc_test_w_avg, acc_test_vote, acc_test_w_vote = self.eval_pdsets(test_pdsets, feat_shape, coor_shape, score_shape)
            _, _, _, _, _, _, _, acc_avg, acc_w_avg, acc_vote, acc_w_vote = self.eval_pdsets(val_pdsets, feat_shape, coor_shape, score_shape)
            _, _, _, _, _, _, _, acc_test_avg, acc_test_w_avg, acc_test_vote, acc_test_w_vote = self.eval_pdsets(test_pdsets, feat_shape, coor_shape, score_shape)

            bool_save = False
            if train_loader is not None:
                if acc_train >= self.best_metric_train:
                    if epoch % 25 == 0:
                        self.best_metric_train = acc_train
                        bool_save = True
            
            if acc_avg >= self.best_metric:
                self.best_metric = acc_avg
                bool_save = True
            
            if acc_test_avg >= self.best_metric_test:
                self.best_metric_test = acc_test_avg 
                bool_save = True

            if train_loader is not None:
                if acc_avg+acc_train >= self.best_metric+self.best_metric_train:
                    bool_save = True
            
            if train_loader is not None:
                if acc_train_w >= self.best_metric_train_weighted:
                    if epoch % 25 == 0:
                        self.best_metric_train_weighted = acc_train_w
                        bool_save = True

            if acc_w_avg >= self.best_metric_weighted:
                self.best_metric_weighted = acc_w_avg
                bool_save = True

            if acc_test_w_avg >= self.best_metric_test_weighted:
                self.best_metric_test_weighted = acc_test_w_avg
                bool_save = True
                    
            if train_loader is not None:
                if acc_w_avg+acc_train_w >= self.best_metric_weighted+self.best_metric_train_weighted:
                    bool_save = True

            if acc_avg > self.best_metric - 0.0125:
                bool_save = True
            if acc_w_avg > self.best_metric_weighted - 0.0125:
                bool_save = True

            if bool_save:
                self.save(epoch, "epoch[%05d]_tr[%.3f]_va[%.3f]_te[%.3f]_trW[%.3f]_vaW[%.3f]_teW[%.3f]"
                    % (epoch, acc_train, acc_avg, acc_test_avg, acc_train_w, acc_w_avg, acc_test_w_avg))
                
            if epoch%25 == 0:
                print('best validation acc: {:.4f}\nbest test acc: {:.4f}'.format(self.best_metric, self.best_metric_test))

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    
    dataset = worker_info.dataset
    # overall_start = dataset.start
    # overall_end = dataset.end
    #per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    # dataset.start = overall_start + worker_id * per_worker
    # dataset.end = min(dataset.start + per_worker, overall_end)
    
    split_size = len(dataset.data) // worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size: (worker_id + 1) * split_size]