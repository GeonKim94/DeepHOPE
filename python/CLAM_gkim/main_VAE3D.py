import math
import os
import pathlib

import numpy as np
import h5py
import argparse
import scipy.io as io

import torch
import torch.nn as nn

from models.VAE3D import VAE3D,vae_loss,mse_loss,kld_loss
# torch.backends.cudnn.benchmark = True

# example for mnist
from datas.Patch3DLoader import Patch3DLoader

import utils
from datas.preprocess3d import TRAIN_AUGS_3D, TEST_AUGS_3D, TRAIN_NOAUGS_3D

from utils.Logger import Logger
from utils.loss import Loss_PCC

from models.VAE3D import VAE3D

import itertools

"""parsing and configuration"""


def arg_parse():
    # projects description
    desc = "3D VAE for unsupervised tomogram 3D patch feature extraction"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', type=str, help="The Directory of data path.")
    parser.add_argument('--gpus', type=str, default="3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Select CPU Number workers")
    parser.add_argument('--aug', type=float, default=1, help='The number of Augmentation Rate')
                        
    parser.add_argument('--load_fname',    type=str, default=None)

    parser.add_argument('--model', type=str, default='hydense',
                        choices=["VAE3D",],
                        help='The type of Models | VAE3D |')
    parser.add_argument('--latent_dim', type=int, default=8096, help='The number of latent features')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')
    
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory name to save the inference result')
                  
    parser.add_argument('--w_loss', nargs="*", type=float, default=(1., 1., 1.)) # weight for validation metric

    parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    
    parser.add_argument('--test', action="store_true", help='Only Test')
    parser.add_argument('--itertools', action="store_true", help='Use itertool to enumerate through data faster (no shuffle)')

    parser.add_argument('--optim', type=str, default='adam', choices=["adam", "sgd"])
    parser.add_argument('--lr', type=float, default=0.001)
    # Adam Optimizer
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))
    # SGD Optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=1e-4)

    return parser.parse_args()


# Define the VAE loss function
def vae_loss(reconstruction, x, mu, logvar):
    BCE = nn.BCELoss(reduction='mean')(reconstruction, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(BCE.item())
    print(KLD.item())
    return BCE+KLD

# Define the MSE loss function
def mse_loss(recon_x, x):
    return nn.MSELoss(reduction='mean')(recon_x, x)

# Define the KLD loss function
def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# save model state_dict
def save_model(path,epoch,net,optimizer,model_type,best_mse,best_pcc,best_sum):
    torch.save({"model_type": model_type,
                "start_epoch": epoch + 1,
                "network": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mse": best_mse,
                "best_pcc": best_pcc,
                "best_sum": best_sum,
                }, path)
    print("Model saved %d epoch" % (epoch))

def load_model(path,net,optimizer):
    """ Model load. same with save"""
    if path is None:
        # load last epoch model
        paths = sorted(glob(arg.save_dir + "/*.pth.tar"))
        print(len(paths))
        if len(paths) == 0:
            print("Not Load")
            return net, optimizer, math.inf, math.inf, math.inf
        else:
            path = os.path.basename(paths[-1]) # load the last model if not really 

    if os.path.exists(path) is True:
        print("Load %s to %s File" % (arg.save_dir, os.path.basename(path)))
        ckpoint = torch.load(path)
        if ckpoint["model_type"] != arg.model:
            raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

        net.load_state_dict(ckpoint['network'])
        optimizer.load_state_dict(ckpoint['optimizer'])
        start_epoch = ckpoint['start_epoch']
        best_mse = ckpoint["best_mse"]
        best_pcc = ckpoint["best_pcc"]
        best_sum = ckpoint["best_sum"]
        print("Load Model Type : %s" % (os.path.basename(path)))
    else:
        start_epoch = 0
        best_mse = math.inf
        best_pcc = math.inf
        best_sum = math.inf
        print("Load Failed, not exists file")
    return start_epoch, net, optimizer, best_mse, best_pcc, best_sum
def create_folder_with_parents(folder_path):
    try:
        # Create the folder and its parent directories if they don't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except OSError as e:
        print(f"Error creating folder '{folder_path}': {e}")
def get_folder_paths(path):
    # Split the path into directory and file components
    directory, filename = os.path.split(path)
    
    # Split the directory into a list of folder names
    folder_names = directory.split(os.path.sep)
    
    # Filter out any empty strings
    folder_names = [folder for folder in folder_names if folder]
    
    # Generate the full path of each folder
    folder_paths = [os.path.join(directory, folder) for folder in folder_names]

    return folder_paths

if __name__ == "__main__":
    arg = arg_parse()

    if arg.result_dir is None:
        arg.result_dir = arg.save_dir    
        
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    load_fname = arg.load_fname
    
    logger = Logger(arg.save_dir)
    logger.will_write(str(arg) + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    torch.cuda.current_device()  # device flagship

    data_path = arg.data_dir
    
    if arg.test is False:
        train_loader = Patch3DLoader(data_path + "/train", arg.batch_size, sampler=False,
                                    transform=TRAIN_AUGS_3D, aug_rate=arg.aug,
                                    num_workers=arg.cpus, shuffle=True, drop_last=False)  
    else:
        train_loader = Patch3DLoader(data_path + "/train", arg.batch_size,
                                transform=TEST_AUGS_3D, aug_rate=0,
                                num_workers=arg.cpus, shuffle=False, drop_last=False)

    val_loader = Patch3DLoader(data_path + "/val", arg.batch_size,
                                transform=TEST_AUGS_3D, aug_rate=0,
                                num_workers=arg.cpus, shuffle=False, drop_last=False)
    test_loader = Patch3DLoader(data_path + "/test", arg.batch_size,
                                transform=TEST_AUGS_3D, aug_rate=0,
                                num_workers=arg.cpus, shuffle=False, drop_last=False)

    
    pcc_loss = Loss_PCC(eps = 1e-6, torch_device = torch_device)

    print("dataset formulated")
    net = VAE3D(input_channels=1, latent_dim=arg.latent_dim)
    net = nn.DataParallel(net).to(torch_device)
    print("network formed")

    optimizer = {
        "adam": torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
        "sgd": torch.optim.SGD(net.parameters(),
                               lr=arg.lr, momentum=arg.momentum,
                               weight_decay=arg.decay, nesterov=True)
    }[arg.optim]


        
    best_mse = math.inf
    best_pcc = math.inf
    best_sum = math.inf
    start_epoch = 0



    if arg.test is False:
        
        if arg.load_fname != None:
            start_epoch, net, optimizer, best_mse, best_pcc, best_kld = load_model(arg.save_dir+'/'+arg.load_fname,net,optimizer)     

        if arg.itertools:
            enum_train = itertools.cycle(train_loader)
            enum_test = itertools.cycle(test_loader)
            enum_val = itertools.cycle(val_loader)

            for ii in range(start_epoch,arg.epoch):
                    #train
                net.train()
                print('starting to train epoch[%05d]' % ii)
                batch_current = 0
                for (input_, target_, path) in enum_train:
                    input_, target_ = input_.to(torch_device), target_.to(torch_device)
                    optimizer.zero_grad()
                    recon, mu, logvar = net(input_)
                    #loss = vae_loss(recon, input_data, mu, logvar).to(torch_device)
                    mse = mse_loss(recon, input_)
                    kld = kld_loss(mu, logvar)
                    pcc = pcc_loss(recon, input_)
                    
                    loss = arg.w_loss[0]*mse+arg.w_loss[1]*pcc+arg.w_loss[2]*kld
                    loss.backward()
                    optimizer.step()
                    batch_current = batch_current+1
                    print('\r')
                    print("training epoch[%05d]: %d/%d" % (ii, batch_current, len(train_loader)))
                    if batch_current == len(train_loader):
                        break

                    #validate
                net.eval()
                saved = False

                mse_avg = 0
                pcc_avg = 0
                kld_avg = 0
                batch_current = 0
                for (input_, target_, path) in enum_val:
                    input_, target_ = input_.to(torch_device), target_.to(torch_device)
                    recon, mu, logvar = net(input_)
                    #loss = vae_loss(recon, input_data, mu, logvar).to(torch_device)
                    mse = mse_loss(recon, input_)
                    kld = kld_loss(mu, logvar)
                    pcc = pcc_loss(recon, input_)
                    
                    mse_avg += mse * input_.shape[0]
                    pcc_avg += pcc * input_.shape[0]
                    kld_avg += kld * input_.shape[0]
                    batch_current = batch_current+1
                    if batch_current == len(val_loader):
                        break
                mse_avg = mse_avg/len(val_loader.dataset)
                pcc_avg = pcc_avg/len(val_loader.dataset)
                kld_avg = kld_avg/len(val_loader.dataset)
                
                mse_avg = mse_avg.detach().cpu().numpy().tolist()
                pcc_avg = pcc_avg.detach().cpu().numpy().tolist()
                kld_avg = kld_avg.detach().cpu().numpy().tolist()

                print("epoch[%05d] mse = %.4f, pcc = %.4f kld = %.4f" % (ii, mse_avg, pcc_avg, kld_avg))
                
                if pcc_avg > 0.4: #% to prevent early saving

                    if ii%50 == 0:
                        pass
                    else:
                        saved = True

                if pcc_avg > best_pcc + 0.01:
                    saved = True

                if mse_avg < best_mse:
                    best_mse = mse_avg
                    if saved:
                        pass
                    else:
                        fname_save = "epoch[%05d]_mse[%.4f]_pcc[%.4f]_kld[%.4f].pth.tar" % (ii, mse_avg, pcc_avg, kld_avg)
                        save_model(arg.save_dir+'/'+fname_save,ii,net,optimizer,arg.model,best_mse,best_pcc,best_sum)
                        saved = True
                if pcc_avg < best_pcc:
                    best_pcc = pcc_avg
                    if saved:
                        pass
                    else:
                        fname_save = "epoch[%05d]_mse[%.4f]_pcc[%.4f]_kld[%.4f].pth.tar" % (ii, mse_avg, pcc_avg, kld_avg)
                        save_model(arg.save_dir+'/'+fname_save,ii,net,optimizer,arg.model,best_mse,best_pcc,best_sum)
                        saved = True
                if mse_avg + pcc_avg < best_sum:
                    best_sum = pcc_avg + mse_avg
                    if saved:
                        pass
                    else:
                        fname_save = "epoch[%05d]_mse[%.4f]_pcc[%.4f]_kld[%.4f].pth.tar" % (ii, mse_avg, pcc_avg, kld_avg)
                        save_model(arg.save_dir+'/'+fname_save,ii,net,optimizer,arg.model,best_mse,best_pcc,best_sum)
                        saved = True
                    logger.log_write("valid", loss_mse=mse_avg, loss_pcc=pcc_avg, loss_kld=kld_avg)

        else: 
            for ii in range(start_epoch,arg.epoch):
                    #train
                net.train()
                print('starting to train epoch[%05d]' % ii)
                batch_current = 0
                for i, (input_, target_, path) in enumerate(train_loader):
                    input_, target_ = input_.to(torch_device), target_.to(torch_device)
                    optimizer.zero_grad()
                    recon, mu, logvar = net(input_)
                    #loss = vae_loss(recon, input_data, mu, logvar).to(torch_device)
                    mse = mse_loss(recon, input_)
                    kld = kld_loss(mu, logvar)
                    pcc = pcc_loss(recon, input_)
                    
                    loss = arg.w_loss[0]*mse+arg.w_loss[1]*pcc+arg.w_loss[2]*kld
                    loss.backward()
                    optimizer.step()
                    batch_current = batch_current+1
                    print('\r')
                    print("training epoch[%05d]: %d/%d" % (ii, batch_current, len(train_loader)))

                    #validate
                net.eval()
                saved = False

                mse_avg = 0
                pcc_avg = 0
                kld_avg = 0
                batch_current = 0
                for i, (input_, target_, path) in enumerate(val_loader):
                    input_, target_ = input_.to(torch_device), target_.to(torch_device)
                    recon, mu, logvar = net(input_)
                    #loss = vae_loss(recon, input_data, mu, logvar).to(torch_device)
                    mse = mse_loss(recon, input_)
                    kld = kld_loss(mu, logvar)
                    pcc = pcc_loss(recon, input_)
                    
                    mse_avg += mse * input_.shape[0]
                    pcc_avg += pcc * input_.shape[0]
                    kld_avg += kld * input_.shape[0]
                    batch_current = batch_current+1
                mse_avg = mse_avg/len(val_loader.dataset)
                pcc_avg = pcc_avg/len(val_loader.dataset)
                kld_avg = kld_avg/len(val_loader.dataset)
                
                mse_avg = mse_avg.detach().cpu().numpy().tolist()
                pcc_avg = pcc_avg.detach().cpu().numpy().tolist()
                kld_avg = kld_avg.detach().cpu().numpy().tolist()

                print("epoch[%05d] mse = %.4f, pcc = %.4f kld = %.4f" % (ii, mse_avg, pcc_avg, kld_avg))

                if pcc_avg > 0.4: #% to prevent early saving

                    if ii%50 == 0:
                        pass
                    else:
                        saved = True

                if pcc_avg > best_pcc + 0.01:
                    saved = True

                if mse_avg < best_mse:
                    best_mse = mse_avg
                    if saved:
                        pass
                    else:
                        fname_save = "epoch[%05d]_mse[%.4f]_pcc[%.4f]_kld[%.4f].pth.tar" % (ii, mse_avg, pcc_avg, kld_avg)
                        save_model(arg.save_dir+'/'+fname_save,ii,net,optimizer,arg.model,best_mse,best_pcc,best_sum)
                        saved = True
                if pcc_avg < best_pcc:
                    best_pcc = pcc_avg
                    if saved:
                        pass
                    else:
                        fname_save = "epoch[%05d]_mse[%.4f]_pcc[%.4f]_kld[%.4f].pth.tar" % (ii, mse_avg, pcc_avg, kld_avg)
                        save_model(arg.save_dir+'/'+fname_save,ii,net,optimizer,arg.model,best_mse,best_pcc,best_sum)
                        saved = True
                if mse_avg + pcc_avg < best_sum:
                    best_sum = pcc_avg + mse_avg
                    if saved:
                        pass
                    else:
                        fname_save = "epoch[%05d]_mse[%.4f]_pcc[%.4f]_kld[%.4f].pth.tar" % (ii, mse_avg, pcc_avg, kld_avg)
                        save_model(arg.save_dir+'/'+fname_save,ii,net,optimizer,arg.model,best_mse,best_pcc,best_sum)
                        saved = True
                    logger.log_write("valid", loss_mse=mse_avg, loss_pcc=pcc_avg, loss_kld=kld_avg)
    else:
        train_loader = Patch3DLoader(data_path + "/train", 1, sampler=False,
                                    transform=TRAIN_NOAUGS_3D, aug_rate=0,
                                    num_workers=arg.cpus, shuffle=False, drop_last=False)  
        val_loader = Patch3DLoader(data_path + "/val", 1,
                                    transform=TEST_AUGS_3D, aug_rate=0,
                                    num_workers=arg.cpus, shuffle=False, drop_last=False)
        test_loader = Patch3DLoader(data_path + "/test", 1,
                                    transform=TEST_AUGS_3D, aug_rate=0,
                                    num_workers=arg.cpus, shuffle=False, drop_last=False)
        
        idx_comma = [index for index, value in enumerate(arg.load_fname) if value == ',']
        #idx_comma = arg.load_fname.index(',')
        idx_front = [-1]
        idx_front = idx_front+idx_comma
        idx_comma.append(len(arg.load_fname))
        
        if len(idx_front) > 1:
            for order_model in range(len(idx_comma)):
                fname_temp = arg.load_fname[idx_front[order_model]+1:idx_comma[order_model]]  
                
                start_epoch, net, optimizer, best_mse, best_pcc, best_kld = load_model(arg.save_dir+'/'+fname_temp,net,optimizer)
                net.eval()
                for i, (input_, target_, path_) in enumerate(train_loader):
                    input_, target_ = input_.to(torch_device), target_.to(torch_device)
                    recon, mu, logvar = net(input_)

                    recon, mu, logvar = recon.detach().cpu().numpy(), mu.detach().cpu().numpy(), logvar.detach().cpu().numpy()

                    
                    path_ = path_[0] ### only works for batch size 1 for testing
                    path_save_recon = path_.replace(arg.data_dir, arg.result_dir+fname_temp.replace('.pth.tar','')+'/recon/')#.replace(pathlib.Path(path_),'h5')
                    path_save_feat = path_.replace(arg.data_dir, arg.result_dir+fname_temp.replace('.pth.tar','')+'/feat/')#.replace(pathlib.Path(path_),'h5')
                    
                    create_folder_with_parents(os.path.dirname(path_save_recon))
                    create_folder_with_parents(os.path.dirname(path_save_feat))

                    h5f_recon = h5py.File(path_save_recon, 'w')
                    h5f_feat = h5py.File(path_save_feat, 'w')

                    h5f_recon.create_dataset('recon', data = recon)
                    h5f_feat.create_dataset('mu', data = mu)
                    h5f_feat.create_dataset('logvar', data = logvar)

                    h5f_recon.close()
                    h5f_feat.close()

        else:
            fname_temp = arg.load_fname
                
            start_epoch, net, optimizer, best_mse, best_pcc, best_kld = load_model(arg.save_dir+'/'+fname_temp,net,optimizer)
            net.eval()
            for i, (input_, target_, path_) in enumerate(train_loader):
                input_, target_ = input_.to(torch_device), target_.to(torch_device)
                recon, mu, logvar = net(input_)

                recon, mu, logvar = recon.cpu().numpy(), mu.cpu().numpy(), logvar.cpu().numpy()

                path_save_recon = path_.replace(arg.data_dir, arg.result_dir+'/'+fname_temp.replace('.pth.tar','')+'/recon/').replace(pathlib.Path(path_),'h5')
                path_save_feat = path_.replace(arg.data_dir, arg.result_dir+'/'+fname_temp.replace('.pth.tar','')+'/feat/').replace(pathlib.Path(path_),'h5')
                
                h5f_recon = h5py.File(path_save_recon, 'w')
                h5f_feat = h5py.File(path_save_feat, 'w')

                h5f_recon.create_dataset('recon', data = recon)
                h5f_feat.create_dataset('mu', data = mu)
                h5f_feat.create_dataset('logvar', data = logvar)

                h5f_recon.close()
                h5f_feat.close()

