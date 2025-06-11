import torch
from scipy.io import savemat
from os.path import basename, splitext, isfile, isdir, dirname, join, getmtime
from os import makedirs
from torch.utils.tensorboard import SummaryWriter
import math
import glob
from pathlib import Path

def check_device_consistency(model, optimizer):
    model_device = next(model.parameters()).device  # Get model parameter device
    print(f"Model device: {model_device}")

    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            print(f"Optimizer parameter device: {param.device}")
            if param.device != model_device:
                print("Mismatch found!")

    print("All optimizer states should be on the same device as the model parameters.")
class runner_imgcls():
    def __init__(self, net, device, optim, sched, loss, dir_ckpt, fname_ckpt, dir_infer, verbose):
        self.device = device
        self.net = net
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.to(self.device)
        self.epoch = 0
        self.optim = optim
        self.sched = sched
        self.loss = loss
        self.dir_ckpt = dir_ckpt
        self.dir_infer = dir_infer
        self.loss_val_best = math.inf
        self.verbose = verbose
        self.load_ckpt(dir_ckpt+'/'+fname_ckpt)
        self.logger = SummaryWriter(log_dir=self.dir_ckpt+'/log_writer')

    def load_ckpt(self, path_ckpt):
        if not isfile(path_ckpt):
            paths_ckpt = glob.glob(join(self.dir_ckpt, "*.pth.tar"))
            if not paths_ckpt:
                print(f'No ckpt file found under {self.dir_ckpt}')
                return
            path_latest = max(paths_ckpt, key=getmtime)
            print(f'Requested {basename(path_ckpt)} not found, loading latest {basename(path_latest)} instead')
            dict_ckpt = torch.load(path_latest)
        else:    
            dict_ckpt = torch.load(path_ckpt)
        first_key = next(iter(dict_ckpt['network']))
        bool_ckpt_dp = first_key.startswith("module.")
        if isinstance(self.net, torch.nn.DataParallel):
            if bool_ckpt_dp:
                self.net.load_state_dict(dict_ckpt["network"])
            else:
                net_ckpt_match = {"module." + k: v for k, v in dict_ckpt["network"].items()}
                self.net.load_state_dict(net_ckpt_match)
        else:
            if bool_ckpt_dp:
                net_ckpt_match = {k.replace("module.", ""): v for k, v in dict_ckpt["network"].items()}
                self.net.load_state_dict(net_ckpt_match)
            else:
                self.net.load_state_dict(dict_ckpt["network"])
        self.optim.load_state_dict(dict_ckpt["optimizer"])
        self.loss_val_best = dict_ckpt["best_metric"]
        if self.sched:
            try:
                self.sched.load_state_dict(dict_ckpt["scheduler"])
            except:
                pass
        try:
            self.epoch = dict_ckpt["epoch"]
        except:
            fname_ckpt = basename(path_ckpt)
            idx1 = fname_ckpt.find('epoch[')
            idx2 = fname_ckpt.find(']')
            if idx1 != -1:
                self.epoch = int(fname_ckpt[idx1+6:idx2])
            
        print(f'ckpt loaded from {Path(path_ckpt).name}')

    def save_ckpt(self,loss_val):
        if isinstance(self.net, torch.nn.DataParallel):
            net_state_dict = self.net.module.state_dict()
        else:
            net_state_dict = self.net.state_dict()

        dict_ckpt = {"epoch": self.epoch,
                    "net": net_state_dict,
                    "optim": self.optim.state_dict(),
                    "loss_val_best": self.loss_val_best,
                    "loss_val": loss_val,
                    }
        if self.sched:
            dict_ckpt["sched"] = self.sched.state_dict()
        torch.save(dict_ckpt, self.dir_ckpt + f"/[epoch{self.epoch:05}]_lossval{loss_val:.04f}.pth.tar")
        print("* Checkpoint saved at epoch %d" % (self.epoch))

    def train_repeat(self, loader_train, loader_val, loader_test, epoch_stop):
        for epoch in range(self.epoch+1, epoch_stop+1):
            self.train(loader_train)

            loss_train = None
            loss_val = None
            loss_test = None

            self.epoch += 1
            loss_train = self.infer(loader_train)
            self.logger.add_scalar('Loss/train', loss_train, self.epoch)
            if loader_val:
                loss_val = self.infer(loader_val)
                self.logger.add_scalar('Loss/val', loss_val, self.epoch)
            if loader_test:
                loss_test = self.infer(loader_test)
                self.logger.add_scalar('Loss/test', loss_test, self.epoch)
            self.logger.flush()

            if self.verbose:
                print(f"[epoch{epoch:05}] ", end = "")
                print(f"Training set loss: {loss_train:.04}", end = " / ")
                if loss_val:
                    print(f"Validation set loss: {loss_val:.04}", end = " / ")
                if loss_test:
                    print(f"Test set loss: {loss_test:.04}", end = " ")

            if loss_val:
                if loss_val < self.loss_val_best*1.01:
                    self.save_ckpt(loss_val)
                else:
                    print("\n")
                
                if loss_val < self.loss_val_best:
                    self.loss_val_best = loss_val
            else:
                self.save_ckpt(loss_val)
        self.logger.close()
    
    def train(self, loader):
        self.net.train()
        for i, (input_b, target_b, _) in enumerate(loader):
            input_b, target_b = input_b.to(self.device), target_b.to(self.device)
            self.optim.zero_grad()
            output_b = self.net(input_b)
            loss_b = self.loss(output_b,target_b)
            loss_b.backward()
            self.optim.step()
            if self.sched:
                self.sched.step()

    def infer(self, loader, savefile = False):
        if not loader:
            return 0
        torch.cuda.empty_cache()
        self.net.eval()
        loss_current = 0
        for i, (input_b, target_b, path_b) in enumerate(loader):
            with torch.no_grad():
                input_b, target_b = input_b.to(self.device), target_b.to(self.device)
                output_b = self.net(input_b)
                
                if savefile:
                    pred_b = torch.argmax(output_b, dim=1)
                    idxs_wrong = torch.where(pred_b != target_b)[0]
                    if self.verbose:
                        for idx_wrong in idxs_wrong:
                            print(f'{basename(path_b[idx_wrong])} misclassified as {loader.dataset.classnames[pred_b[idx_wrong]]} (GT: {loader.dataset.classnames[target_b[idx_wrong]]})')
                
                loss_current += self.loss(output_b,target_b)*input_b.shape[0]
                if savefile:
                    makedirs(self.dir_infer + f"/epoch[{self.epoch:05}]/", exist_ok = True)
                    for iter_data in range(len(path_b)):
                        dict_save = dict()
                        dict_save['path'] = path_b[iter_data]
                        dict_save['input'] = input_b[iter_data].cpu().numpy()
                        dict_save['target'] = target_b[iter_data].cpu().numpy()
                        dict_save['output'] = output_b[iter_data].cpu().numpy()
                        fname_save = splitext(basename(path_b[iter_data]))[0]
                        subdir_save = basename(dirname(path_b[iter_data]))
                        path_save = self.dir_infer + f"/epoch[{self.epoch:05}]/{subdir_save}/{fname_save}.mat"
                        makedirs(dirname(path_save),exist_ok = True)
                        savemat(path_save,
                                dict_save)
        return loss_current/len(loader.dataset)
