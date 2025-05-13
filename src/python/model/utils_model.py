from .resnet.resnet_patch_2d import ResNet_patch
from .resnet.mresnet_patch_2d import mResNet_patch

def get_model_imgcls(type_model,ch_in,num_class,img_size,aug_arch):
    if type_model == "resnet":
        model = ResNet_patch(ch_in,
                             aug_arch['ch_ft'],
                             aug_arch['growth_rate'],
                             aug_arch['block'],
                             aug_arch['layers'],
                             num_class, 
                             img_size,
                             aug_arch['att_type'],
                             aug_arch['gap'])
    if type_model == "mresnet":
        model = mResNet_patch(ch_in,
                             aug_arch['ch_ft'],
                             aug_arch['growth_rate'],
                             aug_arch['block'],
                             aug_arch['layers'],
                             num_class, 
                             img_size,
                             aug_arch['att_type'],
                             aug_arch['gap'])
    return model