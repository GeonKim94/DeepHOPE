import os
import torch
from vit_pytorch import SimpleViT
from vit_pytorch.extractor import Extractor
from datas.TomoLoader import TomoLoader
from runners.clsRunner import clsRunner
import datas.preprocess3d as pp3d # TRAIN_AUGS_3D, TEST_AUGS_3D, crop_shape, size_z
import datas.preprocess25d as pp25d # #TRAIN_AUGS_25D, TEST_AUGS_25D, TRAIN_AUGS_25D_v4, TEST_AUGS_25D_v4
import datas.preprocess2d as pp2d # TRAIN_AUGS_2D, TEST_AUGS_2D


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
torch_device = torch.device("cuda")

image_size = 3072#4096
depth_z = 12
patch_size = 256#256
num_classes = 10
pats_exclude = ('230427',)+('230713',)+('230714',)+('230425',)+('H9',)+('h9',)+('JAX',)+('jax',)
pats_class = ('02_ecto_06h',)+('02_ecto_12',)+('02_ecto_24h',)+('03_meso_06h',)+('03_meso_12',)+('03_meso_24h',)+('04_endo_06h',)+('04_endo_12',)+('04_endo_24h',)+('05_ctl',)
mode_class = 0
reset_class = False
w_metric = (10., 10., 10., 10., 10., 10., 10., 10., 10., 1., )

pad25d = lambda img: pp25d.pad_25d(img, (image_size, image_size))
cencrop25d = lambda img: pp25d.center_crop_25d_alt(img, (image_size,image_size,depth_z))
tform_test = [
    pad25d,
    cencrop25d,
    pp25d.calibration,
    pp25d.channel_fromz,
    pp25d.to_tensor
]

randcrop25d = lambda img: pp25d.random_crop_25d_alt(img, (image_size,image_size,depth_z))
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
    

path_data = '/workspace01/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA/'
train_loader = TomoLoader(path_data+'/train/', 16, 
                            transform=tform_train, aug_rate=0,
                            num_workers=4, shuffle=False, drop_last=True,
                            pats_exclude = pats_exclude,pats_class = pats_class,
                            reset_class = reset_class, mode_class = mode_class)
val_loader = TomoLoader(path_data+'/val/', 16, 
                            transform=tform_train, aug_rate=0,
                            num_workers=4, shuffle=False, drop_last=True,
                            pats_exclude = pats_exclude,pats_class = pats_class,
                            reset_class = reset_class, mode_class = mode_class)
test_loader = TomoLoader(path_data+'/test/', 16, 
                            transform=tform_train, aug_rate=0,
                            num_workers=4, shuffle=False, drop_last=True,
                            pats_exclude = pats_exclude,pats_class = pats_class,
                            reset_class = reset_class, mode_class = mode_class)


net = SimpleViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = num_classes,
    dim = 512,
    depth = 3,#6 for normal
    heads = 1,
    mlp_dim = 512,
    channels = 12
)




model_type = 'SimpleViT'
epoch = 1000
save_dir = '/workspace01/gkim/stem_cell_jwshin/outs/240718_VIT_3072_256'
result_dir = '/workspace01/gkim/stem_cell_jwshin/outs/240718_VIT_3072_256'

net = net.to(torch_device)


if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

loss = torch.nn.CrossEntropyLoss(weight = torch.Tensor(w_metric).to(torch_device))
optim = torch.optim.Adam(net.parameters())
model = clsRunner(net, optim, torch_device, loss,#logger = logger,
                    model_type = model_type, epoch = epoch, save_dir = save_dir, result_dir = result_dir, 
                    w_metric = w_metric, w_metric_test = w_metric, w_metric_train = w_metric, n_class = num_classes)

#import pdb; pdb.set_trace()
img, target, path = next(iter(train_loader))
print(net(img.to(torch_device)).shape)

model.train(train_loader, val_loader, test_loader)

