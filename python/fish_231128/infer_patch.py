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
from datas.preprocess3d import TRAIN_AUGS_3D, TEST_AUGS_3D
from datas.preprocess25d import TRAIN_AUGS_25D, TEST_AUGS_25D, TRAIN_AUGS_25D_v4, TEST_AUGS_25D_v4
from datas.preprocess2d import TRAIN_AUGS_2D, TEST_AUGS_2D

from Logger import Logger

from models.Densenet3d import d169_3d, d121_3d, d201_3d, dwdense_3d, d264_3d, dhy_3d
#from models.EffiDense3d import ed169_3d
from models.fishnet import fishnet150, fishnet99, fishnetdw3
from models.fishnet import fishnetdw as fishdw_origin

from models.fishnet2 import fishdw2, fishdw, fish150
from models.fishnet2_2d import fishdw2_2d, fishdw_2d, fish150_2d, fishdeep1_2d, fishdeep10_2d
from models.fish_exfuse import fishdw_exfuse
from models.fish_dropmax import fishdw as fish_dropmax
#from runners.TomoRunner import TomoRunner
from runners.TomoRunner_patch import TomoRunner_patch
# from runners.TomoRunner_itertools import TomoRunner

"""parsing and configuration"""


def arg_parse():
    # projects description
    desc = "Tomogram Classifier"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', type=str, help="The Directory of data path.")
    parser.add_argument('--gpus', type=str, default="3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Select CPU Number workers")

    parser.add_argument('--dim', type=str, default='3d',
                        choices=["3d", "2d", "25d"])
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
                        help='Directory name to save the inference')
                        
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    #parser.add_argument('--crop', type=int, default=160, help='The size of image cropping')
    parser.add_argument('--reset_loss', action="store_true", help='reset the best metric for in  model saving')
    parser.add_argument('--reset_class', action="store_true", help='reset the class distribution from the folder and set it to class 0')
    parser.add_argument('--mode_class', type=int, default=1, help='if 1 folders with first two characters are the same class, if 0 each folder is a different class')

    parser.add_argument('--num_class', type=int, default="2",
                        help="number of classes, only considered when using reset_class")
    parser.add_argument('--sampler', action="store_true", help='Weighted Sampler work')

    
    parser.add_argument('--w_metric', nargs="*", type=float, default=(1., 1.)) # weight for validation metric
    parser.add_argument('--w_metric_test', nargs="*", type=float, default=(1., 1.)) # weight for test metric
    parser.add_argument('--w_metric_train', nargs="*", type=float, default=(1., 1.)) # weight for loss

    parser.add_argument('--pats_exclude', nargs="*", type = str, help='String patterns from dataset file to exclude', default = ())


    parser.add_argument('--pats_class', nargs="*", type = str, help='String patterns from dataset file to exclude', default = ())

    parser.add_argument('--itertools', type = bool, help='Use itertool to enumerate through data faster (no shuffle)', default = False)

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
        tform_test = TEST_AUGS_2D
        tform_train = TRAIN_AUGS_2D
    elif arg.dim == "3d":
        tform_test = TEST_AUGS_3D
        tform_train = TRAIN_AUGS_3D
    elif arg.dim == "25d":
        tform_test = TEST_AUGS_25D_v4
        tform_train = TRAIN_AUGS_25D_v4
    
    target_shape = (2048,2048,12)
    datasets_test = TomoPatchLoader.getdatasets(path = os.path.join(arg.data_dir, "test"), patch_size = target_shape,
                        transform = tform_test, test = True, aug_rate = 0,
                        pats_exclude = arg.pats_exclude,pats_class = arg.pats_class,
                        reset_class = arg.reset_class, mode_class = arg.mode_class)

    if arg.dim == "3d":
        depth_ = 1
    elif arg.dim == "25d":
        # datasets_test[2].keys()
        # import pdb;pdb.set_trace()
        for input_, target_, inputDir_, coors_off in (datasets_test[0][0]):
            depth_ = input_.shape[0]
            break
    else: 
        depth_ = 1

    print("dataset formulated")
    if arg.reset_class:
        n_class = arg.num_class
    else:
        #import pdb;pdb.set_trace()
        value_list = [value for value in datasets_test[2].values()]
        value_list = set(value_list)
        n_class = len(value_list)

    print("n_class = {:3}".format(n_class))

    net = get_model(arg, classes=n_class, depth=depth_)
    net = net.to(torch_device)
    
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

    net = nn.DataParallel(net)
    print("network formed")
    loss = nn.CrossEntropyLoss(weight = torch.Tensor(arg.w_metric_train).to(torch_device))

    idx_comma = [index for index, value in enumerate(arg.load_fname) if value == ',']
    idx_front = [-1]
    idx_front = idx_front+idx_comma
    idx_comma.append(len(arg.load_fname))
    
    if len(idx_front) > 1:
        for order_model in range(len(idx_comma)):
            fname_temp = arg.load_fname[idx_front[order_model]+1:idx_comma[order_model]]  
            print(fname_temp)
            model = TomoRunner_patch(arg, net, arg.cpus, datasets_test, torch_device, loss, fname_temp,
                                        arg.w_metric, arg.w_metric_test, arg.w_metric_train, logger, n_class)
            model.test()

    else:
        model = TomoRunner_patch(arg, net, arg.cpus, datasets_test, torch_device, loss, arg.load_fname,
                                    arg.w_metric, arg.w_metric_test, arg.w_metric_train, logger, n_class)
        model.test()
