import torch.nn as nn
from torch.nn import init
from .cbam import *
from .bam import *
from .self_attention import *
from .resblock import BasicBlock, Bottleneck

class mResNet_patch(nn.Module):
    def __init__(self, in_dim, ft_dim, growth_rate, block, layers, num_classes, img_size = 512, att_type=None, gap = False):
        "att_type: BAM, CBAM or SA"
        super(mResNet_patch, self).__init__()
        if block == "Bottleneck":
            self.block = Bottleneck
        else:
            self.block = BasicBlock
        self.att_type = att_type
        self.layers = layers
        self.img_size = img_size

        self.inplanes = ft_dim
        # different model config between ImageNet and CIFAR 

        #self.conv1 = nn.Conv2d(in_dim, ft_dim, kernel_size=11, stride=4, padding=5, bias=False) #
        self.conv1 = nn.Conv2d(in_dim, ft_dim, kernel_size=7, stride=2, padding=3, bias=False) # only difference from wideFOV
        self.bn1 = nn.BatchNorm2d(ft_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #

        if att_type=='BAM':
            self.bam1 = BAM(ft_dim*growth_rate)
            self.bam2 = BAM(ft_dim*growth_rate**2)
            self.bam3 = BAM(ft_dim*growth_rate**3)
        elif att_type == 'SA':

            self.bam1 = SA_module(ft_dim*growth_rate, ft_dim*growth_rate, num_head = 4, compress_qk = 8)

            self.bam2 = SA_module(ft_dim*growth_rate**2, ft_dim*growth_rate**2, num_head = 4, compress_qk = 8)

            self.bam3 = SA_module(ft_dim*growth_rate**3, ft_dim*growth_rate**3, num_head = 4, compress_qk = 8)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None
        
        self.layer1 = self._make_layer(self.block, ft_dim//self.block.expansion*growth_rate, layers[0], stride=2, att_type=att_type) # 
        self.layer2 = self._make_layer(self.block, ft_dim//self.block.expansion*growth_rate**2, layers[1], stride=2, att_type=att_type) #
        self.layer3 = self._make_layer(self.block, ft_dim//self.block.expansion*growth_rate**3, layers[2], stride=2, att_type=att_type) #
        self.layer4 = self._make_layer(self.block, ft_dim//self.block.expansion*growth_rate**4, layers[3], stride=2, att_type=att_type) # 

        if gap:
            self.avgpool = nn.AdaptiveAvgPool2d(1)  # 4096 1 1
            self.fc = nn.Linear(int(ft_dim*growth_rate**4), num_classes)
        else:
            self.avgpool = nn.AvgPool2d(4)  # 4096 2 2
            self.fc = nn.Linear(int(ft_dim*growth_rate**4 *self.img_size/2/2**5/4 *self.img_size/2/2**5/4), num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0
   
    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def get_att(self, x, idx_att = 1):
        
        if self.att_type == "CBAM":
            return self.get_att_cbam(x,idx_att)
        elif self.att_type in ["BAM", "SA"]:
            return self.get_att_other(x,idx_att)
        else:
            SyntaxError("the method get_att does not work for models without attention")
    
    def get_att_cbam(self, x,idx_att):
        count_cbam = 0
        if idx_att > 4 or idx_att < 1:
            ValueError("idx_att of get_att_cbam must be between 1 and 4")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        list_layer = [self.layer1, self.layer2, self.layer3, self.layer4]
        for idx_, layers_ in enumerate(list_layer):
            if idx_att == idx_+1:
                for i, layer in enumerate(layers_):
                    if i == self.layers[idx_att-1]-1: 
                        for unit in layer.children():
                            if isinstance(unit, CBAM):
                                return unit._get_att(x)
                            else:
                                x = unit(x)
                    else:
                        x = layer(x)
            x = layers_(x)



    def get_att_other(self, x,idx_att):
        if idx_att > 3 or idx_att < 1:
            ValueError("idx_att of get_att_other must be between 1 and 3")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if idx_att == 1:
            return self.bam1._get_att(x)
        x = self.bam1(x)

        x = self.layer2(x)
        if idx_att == 2:
            return self.bam2._get_att(x)
        x = self.bam2(x)

        x = self.layer3(x)
        return self.bam3._get_att(x)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.bn1(x)
        
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x