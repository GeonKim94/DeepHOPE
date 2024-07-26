import os
import argparse
import torch
import torch.nn as nn
import time
# torch.backends.cudnn.benchmark = True

# example for mnist
from datas.TomoLoader import TomoLoader
from datas import TomoPatchLoader

import utils
import datas.preprocess3d as pp3d # TRAIN_AUGS_3D, TEST_AUGS_3D, crop_shape, size_z
import datas.preprocess25d as pp25d # #TRAIN_AUGS_25D, TEST_AUGS_25D, TRAIN_AUGS_25D_v4, TEST_AUGS_25D_v4
import datas.preprocess2d as pp2d # TRAIN_AUGS_2D, TEST_AUGS_2D

from Logger import Logger

from models.Densenet3d import d169_3d, d121_3d, d201_3d, dwdense_3d, d264_3d, dhy_3d
#from models.EffiDense3d import ed169_3d
from models.fishnet import fishnet150, fishnet99, fishnetdw3
from models.fishnet import fishnetdw as fishdw_origin

from models.fishnet2 import fishdw2, fishdw, fish150
from models.fishnet2_2d import fishdw2_2d, fishdw_2d, fish150_2d, fishdeep1_2d, fishdeep10_2d
from models.fish_exfuse import fishdw_exfuse
from models.fish_dropmax import fishdw as fish_dropmax
from runners.TomoRunner import TomoRunner, worker_init_fn
# from runners.TomoRunner_itertools import TomoRunner

import itertools

"""parsing and configuration"""


def arg_parse():
    # projects description
    desc = "Tomogram Classifier"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', type=str, help="The Directory of data path.")
    parser.add_argument('--gpus', type=str, default="6",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Select CPU Number workers")

    parser.add_argument('--dim', type=str, default='3d',
                        choices=["3d", "2d", "25d"])

    parser.add_argument('--aug', type=float, default=0, help='The rate of additional random sampling; THIS IS NOT DATA AUGMENTATION RATE')

    parser.add_argument('--norm', type=str, default='bn',
                        choices=["bn", "in"])

    parser.add_argument('--act', type=str, default='lrelu',
                        choices=["relu", "lrelu", "prelu"])
                        
    parser.add_argument('--load_fname',    type=str, default=None)

    parser.add_argument('--model', type=str, default='hydense',
                        choices=["attvgg", "dense169", "dense121", "dense201", "dwdense", "dense264", "hydense",
                                 "dpn92", "dpn98", "dpn107", "dpn131",
                                 "res18", "res34", "res50",
                                 "shake", "afd169", "sed169",
                                 "fish150", "fish99", "fishdworigin", "fishdw3origin",
                                 "fishdw", "fishdw2", "fishtest", "fishexfuse",
                                 "fishdw_2d", "fishdw2_2d", "fish150_2d",
                                 "fishdeep1", "fishdeep2", # unimplemented
                                 "fishdeep1_2d", "fishdeep2_2d", "fishdeep10_2d"],
                        help='The type of Models | vgg16 | dense | attvgg |')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')
    
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory name to save the model')
                        
    parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    #parser.add_argument('--crop', type=int, default=160, help='The size of image cropping')
    parser.add_argument('--test', action="store_true", help='Only Test')
    parser.add_argument('--reset_loss', action="store_true", help='reset the best metric for in  model saving')
    parser.add_argument('--reset_class', action="store_true", help='reset the class distribution from the folder and set it to class 0')
    parser.add_argument('--mode_class', type=int, default=1, help='if 1 folders with first two characters are the same class, if 0 each folder is a different class')

    parser.add_argument('--num_class', type=int, default="2",
                        help="number of classes, only considered when using reset_class")
    parser.add_argument('--sampler', action="store_true", help='Weighted Sampler work')
    parser.add_argument('--testfile', type=int, default=-1, help='which test file selected')

    
    parser.add_argument('--w_metric', nargs="*", type=float, default=(1., 1.)) # weight for validation metric
    parser.add_argument('--w_metric_test', nargs="*", type=float, default=(1., 1.)) # weight for test metric
    parser.add_argument('--w_metric_train', nargs="*", type=float, default=(1., 1.)) # weight for loss
    
    parser.add_argument('--optim', type=str, default='adam', choices=["adam", "sgd"])
    parser.add_argument('--lr', type=float, default=0.001)
    # Adam Optimizer
    parser.add_argument('--beta', nargs="*", type=float, default=(0.9, 0.999))
    # SGD Optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=1e-4)

    parser.add_argument('--pats_exclude', nargs="*", type = str, help='String patterns from dataset file to exclude', default = ())


    parser.add_argument('--pats_class', nargs="*", type = str, help='String patterns from dataset file to exclude', default = ())

    parser.add_argument('--itertools', action="store_true", help='Use itertool to enumerate through data faster (no shuffle)', default = False)

    parser.add_argument('--patchval', action="store_true", help='Use patch ensemble for validation and test result', default = False)

    parser.add_argument('--size_crop', nargs="*", type = int, default = (2048,2048,12))
    parser.add_argument('--size_pad', type = int, default = 2048)
    parser.add_argument('--rate_overlap',type = float, default = 2.0)


    return parser.parse_args()

def get_model(arg, classes, depth):
    if arg.dim == "25d":
        input_c = depth#8
    else:
        input_c = 1#1
    if arg.model == "dense169":
        net = d169_3d(num_classes=classes, sample_size=64, sample_duration=96, norm=arg.norm, act=arg.act)
    elif arg.model == "dense264":
        net = d264_3d(num_classes=classes, sample_size=64, sample_duration=96, norm=arg.norm, act=arg.act)
    elif arg.model == "dense121":
        net = d121_3d(num_classes=classes, sample_size=64, sample_duration=96)
    elif arg.model == "hydense":
        net = dhy_3d(num_classes=classes, sample_size=64, sample_duration=96)
    elif arg.model == "dense201":
        net = d201_3d(num_classes=classes, sample_size=64, sample_duration=96)
    elif arg.model == "fish150":
        net = fish150(num_classes=classes)
    elif arg.model == "fish99":
        net = fishnet99(num_classes=classes)
    elif arg.model == "fishdw":
        net = fishdw(num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishdw2":
        net = fishdw2(num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishexfuse":
        net = fishdw_exfuse(num_classes=classes)
    elif arg.model == "fishdworigin":
        net = fishdw_origin(num_classes=classes)
    elif arg.model == "fishdw3origin":
        net = fishnetdw3(num_classes=classes)
    elif arg.model == "fish150_2d":
        net = fish150_2d(input_c, num_classes=classes)
    elif arg.model == "fishdw_2d":
        net = fishdw_2d(input_c, num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishdw2_2d":
        net = fishdw2_2d(input_c, num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishdeep1_2d":
        net = fishdeep1_2d(input_c, num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishdeep2_2d":
        net = fishdeep1_2d(input_c, num_classes=classes, norm=arg.norm, act=arg.act)
    elif arg.model == "fishdeep10_2d":
        net = fishdeep10_2d(input_c, num_classes=classes, norm=arg.norm, act=arg.act)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num_conv_layers = sum(1 for layer in net.modules() if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.Conv2d))

    print('number of parameters: {}\nnumber of convolutions: {}'.format(num_params, num_conv_layers))

    return net


if __name__ == "__main__":
    arg = arg_parse()

    if arg.result_dir is None:
        arg.result_dir = arg.save_dir

    
    #arg.save_dir = "%s/outs/%s" % (os.getcwd(), arg.save_dir)
    #arg.save_dir = "data01/dhryu/004_LMJ_monocyte_efficacy/outs/%s" %(arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    load_fname = arg.load_fname
    
    logger = Logger(arg.save_dir)
    logger.will_write(str(arg) + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    torch.cuda.current_device()  # device flagship

    #data_path = "/data01/dhryu/000_AML_APL_cell_line/dataset/good_1415/"
    #data_path = "/home/dhryu/000_AML_APL_cell_line/dataset/good/"  # important (train,valid,test = 8:1:1)
    #data_path = "/home/dhryu/000_mice_leukemia/dataset/03_adjusted/split/"
    #data_path = "/home/dhryu/000_AML_APL_cell_line/dataset/split_binary_circshift_LMJ/"
    #data_path = "/data01/dhryu/003_5WBC_cls_SMC/dataset/005_DL_4types_20200521/"
    #data_path = "/media/bmolNAS/YSKim/Total_classification/02_data_split_v6/"
    data_path = arg.data_dir
    
    if arg.dim == "2d":
        pad2d = lambda img: pp2d.pad_2d(img, (arg.size_pad, arg.size_pad))
        cencrop2d = lambda img: pp2d.center_crop_2d(img, arg.size_crop)
        tform_test = [
            pad2d,
            cencrop2d,
            pp2d.calibration,
            pp2d.to_tensor_2d
        ]

        randcrop2d = lambda img: pp2d.random_crop_2d(img, arg.size_crop)
        tform_train = [
            randcrop2d,
            pp2d.calibration,
            pp2d.gaussian_2d,
            pp2d.flipud_2d,
            pp2d.fliplr_2d,
            pp2d.rotate_2d,
            pp2d.to_tensor_2d,
        ]

    elif arg.dim == "3d":
        pad3d = lambda img: pp3d.pad_3d(img, (arg.size_pad, arg.size_pad, arg.size_crop[-1]))
        cencrop3d = lambda img: pp3d.center_crop_3d(img, arg.size_crop)
        tform_test = [
            pad3d,
            cencrop3d,
            pp3d.calibration,
            pp3d.channel_single,
            pp3d.to_tensor
        ]  

        randcrop3d = lambda img: pp3d.random_crop_3d(img, arg.size_crop)
        tform_train = [
            randcrop3d,
            pp3d.calibration,
            pp3d.gaussian_3d,
            pp3d.flipud_3d,
            pp3d.fliplr_3d,
            pp3d.rotate_3d,
            pp3d.channel_single,
            pp3d.to_tensor,
        ]

    elif arg.dim == "25d":
        pad25d = lambda img: pp25d.pad_25d(img, (arg.size_pad, arg.size_pad))
        cencrop25d = lambda img: pp25d.center_crop_25d_alt(img, arg.size_crop)
        tform_test = [
            pad25d,
            cencrop25d,
            pp25d.calibration,
            pp25d.channel_fromz,
            pp25d.to_tensor
        ]

        randcrop25d = lambda img: pp25d.random_crop_25d_alt(img, arg.size_crop)
        tform_train = [
            randcrop25d,
            pp25d.flipud_3d,
            pp25d.fliplr_3d,
            pp25d.swapaxes_3d,
            pp25d.calibration,
            pp25d.gaussian_3d,
            pp25d.channel_fromz,
            pp25d.to_tensor,
        ]
    
    
    # THIS NEEDS TO CHANGE TO LOADING PROCESS IN THE RUNNER
    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    #     # Use DataParallel if there is more than one GPU
    #     net = nn.DataParallel(net)#.to(torch_device)
    #     print(f"Using DataParallel with {num_gpus} GPUs.")
    # else:
    #     # Move the model to the GPU if available
    #     net = net.to(torch_device)
    #     print("Using a single GPU or CPU.")

    loss = nn.CrossEntropyLoss(weight = torch.Tensor(arg.w_metric_train).to(torch_device))

    
    if arg.test is False:
        train_loader = TomoLoader(data_path + "train", arg.batch_size, sampler=arg.sampler,
                                    transform=tform_train, aug_rate=arg.aug,
                                    num_workers=arg.cpus, shuffle=True, drop_last=True,
                                    pats_exclude = arg.pats_exclude,pats_class = arg.pats_class,
                                    reset_class = arg.reset_class, mode_class = arg.mode_class)  
        
        if arg.patchval:
            val_pdsets, classes_, class_to_idx_ = TomoPatchLoader.getdatasets(path = os.path.join(arg.data_dir, "val"), patch_size = arg.size_crop,
                                                        pad_size = arg.size_pad, rate_overlap = arg.rate_overlap,
                                                        transform = tform_test, test = True, aug_rate = 0,
                                                        pats_exclude = arg.pats_exclude,pats_class = arg.pats_class,
                                                        reset_class = arg.reset_class, mode_class = arg.mode_class)
            test_pdsets, classes_, class_to_idx_ = TomoPatchLoader.getdatasets(path = os.path.join(arg.data_dir, "test"), patch_size = arg.size_crop,
                                                        pad_size = arg.size_pad, rate_overlap = arg.rate_overlap,
                                                        transform = tform_test, test = True, aug_rate = 0,
                                                        pats_exclude = arg.pats_exclude,pats_class = arg.pats_class,
                                                        reset_class = arg.reset_class, mode_class = arg.mode_class)
            
        

       
        else:
            val_loader = TomoLoader(data_path + "val", arg.batch_size,#32, #arg.batch_size, 64 for 4 GPUS is also viable but server is sometimes overloaded!
                                    transform=tform_test, aug_rate=0,
                                    num_workers=arg.cpus, shuffle=False, drop_last=False,
                                    pats_exclude = arg.pats_exclude, pats_class = arg.pats_class,
                                    reset_class = arg.reset_class, mode_class = arg.mode_class)   
        
            test_loader = TomoLoader(data_path + "test", arg.batch_size,#32,#arg.batch_size,
                                    transform=tform_test, aug_rate=0,
                                    num_workers=arg.cpus, shuffle=False, drop_last=False,
                                    pats_exclude = arg.pats_exclude, pats_class = arg.pats_class,
                                    reset_class = arg.reset_class, mode_class = arg.mode_class)

        if arg.dim == "3d":
            depth_ = 1
        elif arg.dim == "25d":
            if arg.patchval:
                #import pdb; pdb.set_trace()
                test_loader = torch.utils.data.DataLoader(test_pdsets[0], 1, num_workers = 0)
                for idx_, item_ in enumerate(test_loader):#enumerate(loader_test):
                    break
                #import pdb; pdb.set_trace()
                depth_ = item_[0].shape[1]
            else:
                input_, target_, inputDir_ = next(iter(test_loader))
                depth_ = input_.shape[1]
        else: 
            depth_ = 1
        print("dataset formulated")
        if arg.reset_class:
            n_class = arg.num_class
        else:
            if arg.patchval:
                value_list = [value for value in class_to_idx_.values()]
            else:
                value_list = [value for value in train_loader.dataset.class_to_idx.values()]
            value_list = set(value_list)
            n_class = len(value_list)

        net = get_model(arg, classes=n_class, depth=depth_)
        print('state dict key number: {}'.format(len(net.state_dict().keys())))
        net = net.to(torch_device)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        print("network formed")

        optim = {
            "adam": torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
            "sgd": torch.optim.SGD(net.parameters(),
                               lr=arg.lr, momentum=arg.momentum,
                               weight_decay=arg.decay, nesterov=True)
        }[arg.optim]

 
        if arg.reset_loss:
            reset_loss = 1
        else:
            reset_loss = 0

        model = TomoRunner(arg, net, optim, torch_device, loss, logger, load_fname, reset_loss = reset_loss,
                            w_metric = arg.w_metric, w_metric_test = arg.w_metric_test, w_metric_train = arg.w_metric_train, n_class = n_class)
        if arg.patchval:
            model.train_patchval(train_loader, val_pdsets, test_pdsets)

        else:
            model.train(train_loader, val_loader, test_loader)

    else:        

        if arg.patchval:
            val_pdsets, classes_, class_to_idx_ = TomoPatchLoader.getdatasets(path = os.path.join(arg.data_dir, "val"), patch_size = arg.size_crop,
                                                        pad_size = arg.size_pad, rate_overlap = arg.rate_overlap,
                                                        transform = tform_test, test = True, aug_rate = 0,
                                                        pats_exclude = arg.pats_exclude,pats_class = arg.pats_class,
                                                        reset_class = arg.reset_class, mode_class = arg.mode_class)
            test_pdsets, classes_, class_to_idx_ = TomoPatchLoader.getdatasets(path = os.path.join(arg.data_dir, "test"), patch_size = arg.size_crop,
                                                        pad_size = arg.size_pad, rate_overlap = arg.rate_overlap,
                                                        transform = tform_test, test = True, aug_rate = 0,
                                                        pats_exclude = arg.pats_exclude,pats_class = arg.pats_class,
                                                        reset_class = arg.reset_class, mode_class = arg.mode_class)
            train_pdsets, classes_, class_to_idx_ = TomoPatchLoader.getdatasets(path = os.path.join(arg.data_dir, "train"), patch_size = arg.size_crop,
                                                        pad_size = arg.size_pad, rate_overlap = arg.rate_overlap,
                                                        transform = tform_train, test = True, aug_rate = 0,
                                                        pats_exclude = arg.pats_exclude,pats_class = arg.pats_class,
                                                        reset_class = arg.reset_class, mode_class = arg.mode_class)
     
            if len(train_pdsets) == 0:
                train_pdsets = None
            if len(val_pdsets) == 0:
                val_pdsets = None
        else:
            train_loader = TomoLoader(data_path + "train", arg.batch_size, sampler=arg.sampler,
                                        transform=tform_train, aug_rate=arg.aug,
                                        num_workers=arg.cpus, shuffle=False, drop_last=False, pats_exclude = arg.pats_exclude, pats_class = arg.pats_class, reset_class = arg.reset_class, mode_class = arg.mode_class)  
            val_loader = TomoLoader(data_path + "val", arg.batch_size,#32, #arg.batch_size, 64 for 4 GPUS is also viable but server is sometimes overloaded!
                                        transform=tform_test, aug_rate=0,
                                        num_workers=arg.cpus, shuffle=False, drop_last=False, pats_exclude = arg.pats_exclude, pats_class = arg.pats_class, reset_class = arg.reset_class, mode_class = arg.mode_class) 
            test_loader = TomoLoader(data_path + "test", arg.batch_size,#32,#arg.batch_size,
                                        transform=tform_test, aug_rate=0,
                                        num_workers=arg.cpus, shuffle=False, drop_last=False, pats_exclude = arg.pats_exclude, pats_class = arg.pats_class, reset_class = arg.reset_class, mode_class = arg.mode_class) 
            
            if len(train_loader) == 0:
                train_loader = None
            if len(val_loader) == 0:
                val_loader = None

        if arg.dim == "3d":
            depth_ = 1
        elif arg.dim == "25d":
            if arg.patchval:
                #import pdb; pdb.set_trace()
                test_loader = torch.utils.data.DataLoader(test_pdsets[0], 1, num_workers = 0)
                for idx_, item_ in enumerate(test_loader):#enumerate(loader_test):
                    break
                #import pdb; pdb.set_trace()
                depth_ = item_[0].shape[1]
            else:
                input_, target_, inputDir_ = next(iter(test_loader))
                depth_ = input_.shape[1]
        else: 
            depth_ = 1
        print("dataset formulated")
        if arg.reset_class:
            n_class = arg.num_class
        else:
            if arg.patchval:
                value_list = [value for value in class_to_idx_.values()]
            else:
                value_list = [value for value in train_loader.dataset.class_to_idx.values()]
            value_list = set(value_list)
            n_class = len(value_list)
        #import pdb; pdb.set_trace()

        net = get_model(arg, classes=n_class, depth=depth_)
        net = net.to(torch_device)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        
        optim = {
            "adam": torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
            "sgd": torch.optim.SGD(net.parameters(),
                            lr=arg.lr, momentum=arg.momentum,
                            weight_decay=arg.decay, nesterov=True)
        }[arg.optim]


        idx_comma = [index for index, value in enumerate(arg.load_fname) if value == ',']
        #idx_comma = arg.load_fname.index(',')
        idx_front = [-1];
        idx_front = idx_front+idx_comma
        idx_comma.append(len(arg.load_fname))
        
        if len(idx_front) > 1:
            if arg.patchval:
                for order_model in range(len(idx_comma)):
                    fname_temp = arg.load_fname[idx_front[order_model]+1:idx_comma[order_model]]  
                    print(fname_temp)
                    model = TomoRunner(arg, net, optim, torch_device, loss, logger, fname_temp,
                                w_metric = arg.w_metric, w_metric_test = arg.w_metric_test, w_metric_train = arg.w_metric_train, n_class = n_class)
                    print("model defined")
                    print(arg.w_metric)
                    #model.valid(epoch = 0, val_loader = val_loader, test_loader = test_loader)
                    model.test_patch(val_pdsets, test_pdsets, train_pdsets)
            else:
                for order_model in range(len(idx_comma)):
                    fname_temp = arg.load_fname[idx_front[order_model]+1:idx_comma[order_model]]  
                    print(fname_temp)
                    model = TomoRunner(arg, net, optim, torch_device, loss, logger, fname_temp,
                                w_metric = arg.w_metric, w_metric_test = arg.w_metric_test, w_metric_train = arg.w_metric_train, n_class = n_class)
                    print("model defined")
                    print(arg.w_metric)
                    #model.valid(epoch = 0, val_loader = val_loader, test_loader = test_loader)
                    model.test(train_loader = train_loader, val_loader = val_loader, test_loader = test_loader)
    
        else:
            if arg.patchval:
                print(arg.load_fname)
                start = time.time()
                model = TomoRunner(arg, net, optim, torch_device, loss, logger, arg.load_fname,
                                w_metric = arg.w_metric, w_metric_test = arg.w_metric_test, w_metric_train = arg.w_metric_train, n_class = n_class)
                end = time.time()
                print("model defined, time elapsed = {}".format(end - start))
                print(arg.w_metric)
                #model.valid(0, val_loader, test_loader)
                model.test_patch(val_pdsets, test_pdsets, train_pdsets)
            else:
        
                print(arg.load_fname)
                start = time.time()
                model = TomoRunner(arg, net, optim, torch_device, loss, logger, arg.load_fname,
                                w_metric = arg.w_metric, w_metric_test = arg.w_metric_test, w_metric_train = arg.w_metric_train, n_class = n_class)
                end = time.time()
                print("model defined, time elapsed = {}".format(end - start))
                print(arg.w_metric)
                #model.valid(0, val_loader, test_loader)
                model.test(train_loader = train_loader, val_loader = val_loader, test_loader = test_loader)
