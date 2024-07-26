
# coding: utf-8

# In[1]:


import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


# In[2]:


import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob


# device = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda")

#dirs = sorted(os.listdir("outs"))
#list(enumerate(dirs))


# In[3]:


#target_dir = dirs[57]
#ckpoint_path = sorted(glob("outs/" + target_dir + "/epoch*"))[-1]
#ckpoint_path
#ckpoint_path = "/data02/gkim/stem_cell_jwshin/outs/220803_3D_b016_lr0.001/epoch[00012]_acc[0.7030]_test[0.7265].pth.tar"

ckpoint_path = "/data02/gkim/stem_cell_jwshin/outs/230407_MIP_b004_in_lr0.001/epoch[00009]_acc[0.8667]_test[0.8667].pth.tar"
# In[4]:


import torch

#from models.fishnet2 import fishdw2, fishdw, fish150
#net = fishdw(num_classes=2, norm="bn", act="lrelu")
from models.fishnet2_2d import fishdw2_2d, fishdw_2d, fish150_2d
net = fishdw_2d(num_classes=3, norm="in", act="lrelu")


net = torch.nn.DataParallel(net)
state = torch.load(ckpoint_path)["network"]
net.load_state_dict(state)
model = net.module
model.to(device)
print("Net Load")


# In[5]:


from visualizes.GradCam2d import GradCam2d
#from visualizes.GradCam import GradCam

from visualizes.GuidedBackpropReLUModel import GuidedBackpropReLUModel
#gb_model = GuidedBackpropReLUModel(model, device=device)

# In[7]:

# from datas.preprocess3d import TRAIN_AUGS_3D, TEST_AUGS_3D
from datas.preprocess2d import TRAIN_AUGS_2D, TEST_AUGS_2D
#from datas.BacLoader import bacLoader
#data_path = "/data2/DW/180930_bac/valid40/"
#test_loader = bacLoader(data_path + "test", 1, task="bac", sampler=False,
#                        transform=TEST_AUGS_3D, aug_rate=0,
#                        num_workers=16, shuffle=False, drop_last=False)
from datas.LymphoLoader import lymphoLoader
data_path = "/data02/gkim/stem_cell_jwshin/data/230407_MIP/"
test_loader = lymphoLoader(data_path + "test", 1,
               transform=TEST_AUGS_2D, aug_rate=0,
               num_workers=16, shuffle=False, drop_last=False)


# In[11]:


import torch.nn.functional as F



name_part = "tail_layer"
idx_layer = 0
for i, (input_, label_, path) in enumerate(test_loader):


    grad_cam = GradCam2d(model, name_part, idx_layer, device=device) #hooks=["features", "transition1"]

    torch.cuda.empty_cache()
    input_.requires_grad_(True)    
    respond_cam0, gdcam0, gdcampp0, predict = grad_cam.multicam(input_, 0)#gunhocam0, 
    respond_cam1, gdcam1, gdcampp1, predict = grad_cam.multicam(input_, 1)#gunhocam1, 
    respond_cam2, gdcam2, gdcampp2, predict = grad_cam.multicam(input_, 2)#gunhocam1, 
    
    #gb_img0 = gb_model(input_, 0)
    #gb_img1 = gb_model(input_, 1)
    img = input_[0, 0].detach().numpy()
    
    label = test_loader.dataset.idx_to_class[label_.item()]
    pred = test_loader.dataset.idx_to_class[predict.item()]
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle("File : %s\n Label : %s, pred : %s"%(path, label, pred),
                 fontsize=16, y=0.61)
    
    
    ## for 2d
    cmap="hot"
    ax = plt.subplot(3, 4, 1)
    ax.set_title("RI")
    plt.imshow(img, cmap = cmap)#img[:, :, zdim],         cmap=cmap)
    ax = plt.subplot(3, 4, 2)
    ax.set_title("respond cam")
    print(img.shape)
    print(respond_cam0.shape)
    print(gdcam0.shape)
    print(gdcampp0.shape)
    
    plt.imshow(respond_cam0, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 3)
    ax.set_title("gradcam")
    plt.imshow(gdcam0, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 4)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp0, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ax = plt.subplot(3, 4, 6)
    ax.set_title("respond cam")
    plt.imshow(respond_cam1, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 7)
    ax.set_title("gradcam")
    plt.imshow(gdcam1, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 8)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp1, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ax = plt.subplot(3, 4, 10)
    ax.set_title("respond cam")
    plt.imshow(respond_cam2, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 11)
    ax.set_title("gradcam")
    plt.imshow(gdcam2, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 12)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp2, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ## for 3d
#    cmap="hot"
#    ax = plt.subplot(2, 5, 1)
#    ax.set_title("RI")
#    plt.imshow(np.max(img,axis = 2), cmap = cmap)#img[:, :, zdim],         cmap=cmap)
#    ax = plt.subplot(2, 5, 2)
#    ax.set_title("respond cam")
#    plt.imshow(np.max(respond_cam0,axis = 2), cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
#    ax = plt.subplot(2, 5, 3)
#    ax.set_title("gradcam")
#    plt.imshow(np.max(gdcam0,axis = 2), cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
#    ax = plt.subplot(2, 5, 4)
#    ax.set_title("gdcampp")
#    plt.imshow(np.max(gdcampp0,axis = 2), cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
#    #ax = plt.subplot(2, 5, 5)
#    #ax.set_title("guided bp")
#    #plt.imshow(np.max(gb_img0,axis = 2), cmap = cmap)#gunhocam[:, :, zdim],    cmap=cmap)
#    
#    ax = plt.subplot(2, 5, 7)
#    ax.set_title("respond cam")
#    plt.imshow(np.max(respond_cam1,axis = 2), cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
#    ax = plt.subplot(2, 5, 8)
#    ax.set_title("gradcam")
#    plt.imshow(np.max(gdcam1,axis = 2), cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
#    ax = plt.subplot(2, 5, 9)
#    ax.set_title("gdcampp")
#    plt.imshow(np.max(gdcampp1,axis = 2), cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
#    #ax = plt.subplot(2, 5, 10)
#    #ax.set_title("guided bp")
#    #plt.imshow(np.max(gb_img1,axis = 2), cmap = cmap)#gunhocam[:, :, zdim],    cmap=cmap)


    save_path = "/data02/gkim/stem_cell_jwshin/outs/230407_MIP_b004_in_lr0.001/epoch[00009]_cams/" + name_part + "{}".format(idx_layer) +  "/" + ("True/" if label == pred else "False/")
    #"/data02/gkim/stem_cell_jwshin/outs/220803_3D_b016_lr0.001/epoch[00012]_cams_" + name_part + "{}".format(idx_layer) +  "/" + ("True/" if label == pred else "False/")
    #print(save_path + file_name)
    file_name = path[0][path[0].find("/test/") + 6:-4] + ".png"
    os.makedirs(os.path.dirname(save_path + file_name), exist_ok=True)
    plt.savefig(save_path + file_name, dpi=fig.dpi)
    
    file_name = path[0][path[0].find("/test/") + 6:]
    dict_data = {'ri':img, 'respcam0':respond_cam0, 'gradcam0':gdcam0, 'gradcampp0':gdcampp0, 
    'respcam1':respond_cam1, 'gradcam1':gdcam1, 'gradcampp1':gdcampp1,
    'respcam2':respond_cam2, 'gradcam2':gdcam2, 'gradcampp2':gdcampp2, 'label':label, 'pred':pred}#'guidedbp0':gb_img0,'guidedbp1':gb_img1, 
    savemat(save_path+file_name, dict_data)
    
    #print(save_path + file_name)
    
    # plt.show()
    plt.close()
    
    del grad_cam
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


name_part = "body_layer"
idx_layer = 2
for i, (input_, label_, path) in enumerate(test_loader):


    grad_cam = GradCam2d(model, name_part, idx_layer, device=device) #hooks=["features", "transition1"]

    torch.cuda.empty_cache()
    input_.requires_grad_(True)    
    respond_cam0, gdcam0, gdcampp0, predict = grad_cam.multicam(input_, 0)#gunhocam0, 
    respond_cam1, gdcam1, gdcampp1, predict = grad_cam.multicam(input_, 1)#gunhocam1, 
    
    #gb_img0 = gb_model(input_, 0)
    #gb_img1 = gb_model(input_, 1)
    img = input_[0, 0].detach().numpy()
    
    label = test_loader.dataset.idx_to_class[label_.item()]
    pred = test_loader.dataset.idx_to_class[predict.item()]
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle("File : %s\n Label : %s, pred : %s"%(path, label, pred),
                 fontsize=16, y=0.61)
    
    ## for 2d
    cmap="hot"
    ax = plt.subplot(3, 4, 1)
    ax.set_title("RI")
    plt.imshow(img, cmap = cmap)#img[:, :, zdim],         cmap=cmap)
    ax = plt.subplot(3, 4, 2)
    ax.set_title("respond cam")
    print(img.shape)
    print(respond_cam0.shape)
    print(gdcam0.shape)
    print(gdcampp0.shape)
    
    plt.imshow(respond_cam0, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 3)
    ax.set_title("gradcam")
    plt.imshow(gdcam0, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 4)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp0, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ax = plt.subplot(3, 4, 6)
    ax.set_title("respond cam")
    plt.imshow(respond_cam1, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 7)
    ax.set_title("gradcam")
    plt.imshow(gdcam1, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 8)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp1, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ax = plt.subplot(3, 4, 10)
    ax.set_title("respond cam")
    plt.imshow(respond_cam2, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 11)
    ax.set_title("gradcam")
    plt.imshow(gdcam2, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 12)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp2, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ## for 3d
#    cmap="hot"
#    ax = plt.subplot(2, 5, 1)
#    ax.set_title("RI")
#    plt.imshow(np.max(img,axis = 2), cmap = cmap)#img[:, :, zdim],         cmap=cmap)
#    ax = plt.subplot(2, 5, 2)
#    ax.set_title("respond cam")
#    plt.imshow(np.max(respond_cam0,axis = 2), cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
#    ax = plt.subplot(2, 5, 3)
#    ax.set_title("gradcam")
#    plt.imshow(np.max(gdcam0,axis = 2), cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
#    ax = plt.subplot(2, 5, 4)
#    ax.set_title("gdcampp")
#    plt.imshow(np.max(gdcampp0,axis = 2), cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
#    #ax = plt.subplot(2, 5, 5)
#    #ax.set_title("guided bp")
#    #plt.imshow(np.max(gb_img0,axis = 2), cmap = cmap)#gunhocam[:, :, zdim],    cmap=cmap)
#    
#    ax = plt.subplot(2, 5, 7)
#    ax.set_title("respond cam")
#    plt.imshow(np.max(respond_cam1,axis = 2), cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
#    ax = plt.subplot(2, 5, 8)
#    ax.set_title("gradcam")
#    plt.imshow(np.max(gdcam1,axis = 2), cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
#    ax = plt.subplot(2, 5, 9)
#    ax.set_title("gdcampp")
#    plt.imshow(np.max(gdcampp1,axis = 2), cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
#    #ax = plt.subplot(2, 5, 10)
#    #ax.set_title("guided bp")
#    #plt.imshow(np.max(gb_img1,axis = 2), cmap = cmap)#gunhocam[:, :, zdim],    cmap=cmap)


    save_path = "/data02/gkim/stem_cell_jwshin/outs/230407_MIP_b004_in_lr0.001/epoch[00009]_cams/" + name_part + "{}".format(idx_layer) +  "/" + ("True/" if label == pred else "False/")
    #"/data02/gkim/stem_cell_jwshin/outs/220803_3D_b016_lr0.001/epoch[00012]_cams_" + name_part + "{}".format(idx_layer) +  "/" + ("True/" if label == pred else "False/")
    #print(save_path + file_name)
    file_name = path[0][path[0].find("/test/") + 6:-4] + ".png"
    os.makedirs(os.path.dirname(save_path + file_name), exist_ok=True)
    plt.savefig(save_path + file_name, dpi=fig.dpi)
    
    file_name = path[0][path[0].find("/test/") + 6:]
    dict_data = {'ri':img, 'respcam0':respond_cam0, 'gradcam0':gdcam0, 'gradcampp0':gdcampp0, 
    'respcam1':respond_cam1, 'gradcam1':gdcam1, 'gradcampp1':gdcampp1,
    'respcam2':respond_cam2, 'gradcam2':gdcam2, 'gradcampp2':gdcampp2, 'label':label, 'pred':pred}#'guidedbp0':gb_img0,'guidedbp1':gb_img1, 
    savemat(save_path+file_name, dict_data)
    
    #print(save_path + file_name)
    
    # plt.show()
    plt.close()
    
    del grad_cam
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



name_part = "head_layer"
idx_layer = 0
for i, (input_, label_, path) in enumerate(test_loader):


    grad_cam = GradCam2d(model, name_part, idx_layer, device=device) #hooks=["features", "transition1"]

    torch.cuda.empty_cache()
    input_.requires_grad_(True)    
    respond_cam0, gdcam0, gdcampp0, predict = grad_cam.multicam(input_, 0)#gunhocam0, 
    respond_cam1, gdcam1, gdcampp1, predict = grad_cam.multicam(input_, 1)#gunhocam1, 
    
    #gb_img0 = gb_model(input_, 0)
    #gb_img1 = gb_model(input_, 1)
    img = input_[0, 0].detach().numpy()
    
    label = test_loader.dataset.idx_to_class[label_.item()]
    pred = test_loader.dataset.idx_to_class[predict.item()]
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle("File : %s\n Label : %s, pred : %s"%(path, label, pred),
                 fontsize=16, y=0.61)
    
    ## for 2d
    cmap="hot"
    ax = plt.subplot(3, 4, 1)
    ax.set_title("RI")
    plt.imshow(img, cmap = cmap)#img[:, :, zdim],         cmap=cmap)
    ax = plt.subplot(3, 4, 2)
    ax.set_title("respond cam")
    print(img.shape)
    print(respond_cam0.shape)
    print(gdcam0.shape)
    print(gdcampp0.shape)
    
    plt.imshow(respond_cam0, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 3)
    ax.set_title("gradcam")
    plt.imshow(gdcam0, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 4)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp0, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ax = plt.subplot(3, 4, 6)
    ax.set_title("respond cam")
    plt.imshow(respond_cam1, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 7)
    ax.set_title("gradcam")
    plt.imshow(gdcam1, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 8)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp1, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ax = plt.subplot(3, 4, 10)
    ax.set_title("respond cam")
    plt.imshow(respond_cam2, cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
    ax = plt.subplot(3, 4, 11)
    ax.set_title("gradcam")
    plt.imshow(gdcam2, cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
    ax = plt.subplot(3, 4, 12)
    ax.set_title("gdcampp")
    plt.imshow(gdcampp2, cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
    
    ## for 3d
#    cmap="hot"
#    ax = plt.subplot(2, 5, 1)
#    ax.set_title("RI")
#    plt.imshow(np.max(img,axis = 2), cmap = cmap)#img[:, :, zdim],         cmap=cmap)
#    ax = plt.subplot(2, 5, 2)
#    ax.set_title("respond cam")
#    plt.imshow(np.max(respond_cam0,axis = 2), cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
#    ax = plt.subplot(2, 5, 3)
#    ax.set_title("gradcam")
#    plt.imshow(np.max(gdcam0,axis = 2), cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
#    ax = plt.subplot(2, 5, 4)
#    ax.set_title("gdcampp")
#    plt.imshow(np.max(gdcampp0,axis = 2), cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
#    #ax = plt.subplot(2, 5, 5)
#    #ax.set_title("guided bp")
#    #plt.imshow(np.max(gb_img0,axis = 2), cmap = cmap)#gunhocam[:, :, zdim],    cmap=cmap)
#    
#    ax = plt.subplot(2, 5, 7)
#    ax.set_title("respond cam")
#    plt.imshow(np.max(respond_cam1,axis = 2), cmap = cmap)#respond_cam[:, :, zdim], cmap=cmap)
#    ax = plt.subplot(2, 5, 8)
#    ax.set_title("gradcam")
#    plt.imshow(np.max(gdcam1,axis = 2), cmap = cmap)#gdcam[:, :, zdim],       cmap=cmap)
#    ax = plt.subplot(2, 5, 9)
#    ax.set_title("gdcampp")
#    plt.imshow(np.max(gdcampp1,axis = 2), cmap = cmap)#gdcampp[:, :, zdim],     cmap=cmap)
#    #ax = plt.subplot(2, 5, 10)
#    #ax.set_title("guided bp")
#    #plt.imshow(np.max(gb_img1,axis = 2), cmap = cmap)#gunhocam[:, :, zdim],    cmap=cmap)


    save_path = "/data02/gkim/stem_cell_jwshin/outs/230407_MIP_b004_in_lr0.001/epoch[00009]_cams/" + name_part + "{}".format(idx_layer) +  "/" + ("True/" if label == pred else "False/")
    #"/data02/gkim/stem_cell_jwshin/outs/220803_3D_b016_lr0.001/epoch[00012]_cams_" + name_part + "{}".format(idx_layer) +  "/" + ("True/" if label == pred else "False/")
    #print(save_path + file_name)
    file_name = path[0][path[0].find("/test/") + 6:-4] + ".png"
    os.makedirs(os.path.dirname(save_path + file_name), exist_ok=True)
    plt.savefig(save_path + file_name, dpi=fig.dpi)
    
    file_name = path[0][path[0].find("/test/") + 6:]
    dict_data = {'ri':img, 'respcam0':respond_cam0, 'gradcam0':gdcam0, 'gradcampp0':gdcampp0, 
    'respcam1':respond_cam1, 'gradcam1':gdcam1, 'gradcampp1':gdcampp1,
    'respcam2':respond_cam2, 'gradcam2':gdcam2, 'gradcampp2':gdcampp2, 'label':label, 'pred':pred}#'guidedbp0':gb_img0,'guidedbp1':gb_img1, 
    savemat(save_path+file_name, dict_data)
    
    
    #print(save_path + file_name)
    
    # plt.show()
    plt.close()
    
    del grad_cam
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

