
# coding: utf-8

# In[1]:


import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")


# In[2]:


import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

dirs = sorted(os.listdir("outs"))
list(enumerate(dirs))


# In[3]:


target_dir = dirs[57]
ckpoint_path = sorted(glob("outs/" + target_dir + "/epoch*"))[-1]
ckpoint_path


# In[4]:


import torch
from models.Densenet3d_nomaxpool import d169_3d
net = d169_3d(num_classes=18, sample_size=64, sample_duration=96,
              norm="bn", act="lrelu", se=[], af=[])

net = torch.nn.DataParallel(net)
state = torch.load(ckpoint_path)["network"]
net.load_state_dict(state)
model = net.module
model.to(device)
print("Net Load")


# In[5]:


from visualizes.GradCam import GradCam
# grad_cam = GradCam.get_layer_name(model)


# In[6]:


# from visualizes.GradCam_torch import GradCamPyTorch as GradCam
from visualizes.GradCam import GradCam

grad_cam = GradCam(model=model, hooks=["features", "transition1"], device=device)
# Make New model variance

from visualizes.GuidedBackpropReLUModel import GuidedBackpropReLUModel
gb_model = GuidedBackpropReLUModel(model=model, device=device)


# In[7]:


from datas.BacLoader import bacLoader
from datas.preprocess3d import TRAIN_AUGS_3D, TEST_AUGS_3D

data_path = "/data2/DW/180930_bac/valid40/"

test_loader = bacLoader(data_path + "test", 1, task="bac", sampler=False,
                        transform=TEST_AUGS_3D, aug_rate=0,
                        num_workers=16, shuffle=False, drop_last=False)


# In[11]:


import torch.nn.functional as F
for i, (input_, label_, path) in enumerate(test_loader):
    input_.requires_grad_(True)    
    respond_cam, gdcam, gdcampp, gunhocam, predict = grad_cam.multicam(input_, None)
    
    img = input_[0, 0].detach().numpy()
    
    label = test_loader.dataset.idx_to_class[label_.item()]
    pred = test_loader.dataset.idx_to_class[predict.item()]
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle("File : %s\n Label : %s, pred : %s"%(path, label, pred),
                 fontsize=16, y=0.61)
    
    zdim = 10; cmap="hot"
    ax = plt.subplot(151)
    ax.set_title("image z:%d"%(zdim))
    plt.imshow(img[:, :, zdim],         cmap=cmap)
    ax = plt.subplot(152)
    ax.set_title("respond cam")
    plt.imshow(respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(153)
    ax.set_title("gradcam")
    plt.imshow(gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(154)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp[:, :, zdim],     cmap=cmap)
    ax = plt.subplot(155)
    ax.set_title("gunhocam")
    plt.imshow(gunhocam[:, :, zdim],    cmap=cmap)

    # /data2/DW/180930_bac/valid40/test/Acinetobacter_baumannii/Tomo_532nm_suppr_1_sp007_timeMarker_001_data_180905.mat
    save_path = "outs/" + target_dir + "/cams/" + ("True/" if label == pred else "False/")
    file_name = path[0][path[0].find("/test/") + 6:] + ".png"
    os.makedirs(os.path.dirname(save_path + file_name), exist_ok=True)
    plt.savefig(save_path + file_name, dpi=fig.dpi)
    print(save_path + file_name)
    
    # plt.show()
    plt.close()
    # cam_img = GradCam.cam_on_image(img, cam_mask)
    
    """
    gb_img = gb_model(input_, index=target_index)
    cam_gb = gb_model.gb_on_cam(cam_mask, gb_img)
    
    plt.figure(figsize=(20, 20))
    plt.suptitle("File : %s\n Label : %s, pred : %s"%(path,
                                                      test_loader.dataset.idx_to_class[label_.item()],
                                                      test_loader.dataset.idx_to_class[predict.item()]),
                 fontsize=16, y=0.61)
    
    plt.subplot(141); plt.imshow(img[:, :, 10], cmap="hot")
    plt.subplot(142); plt.imshow(cam_mask[:, :, 10], cmap="hot")
    
    plt.subplot(143); plt.imshow(gb_img[:, :, 10]);
    plt.subplot(144); plt.imshow(cam_gb[:, :, 10]);
    plt.show()
    """

