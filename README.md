# DeepHOPE (deep-learning-assisted holotomography for pluripotency evaluation)
This repository contains codes, models, and data for demonstrating DeepHOPE, which is an integrated pipeline of holotmographic imaging and deep-learning image recognition for assessing the stemness of an hPSC colony. 

## Download datasets and model before testing
Pre-trained model (gastric dataset) : WIP  
Example training and testing dataset : WIP  

## Installation
```shell
conda env create -f environment.yml
```

```shell
conda activate env_DeepHOPE
``` 

## Set the directories

```shell

dataset                
 ├──  data_test.yaml
 ├──  group1     
 |      ├── colony01_patch01.h5     
 |      ├── colony01_patch02.h5     
 |      ├── ...
 └──  group2     
 |      ├── colony01_patch01.h5     
 |      ├── colony01_patch02.h5     
 |      ├── ...
 ├── ...

model                
 ├──  condition1     
 |      ├── epoch[00048].pth
 |      ├── epoch[00079].pth     
 |      ├── ...
 ├──  condition2     
 |      ├── epoch[00034].pth
 |      ├── ...
 ├── ...

codes
 └── codes downloaded from this repository

```

## Edit the config file to
A config file is a yaml file of the following format, that governs the process of either training (train.py) or inference (infer.py).  
``` shell

# Computation
compute:
  gpus: "0"
  cpus: 4
# Computation
compute:
  gpus: "5"
  cpus: 2

# Optimizer
optim:
  loss: "CE"
  batch_size: 2
  epoch: 200
  optimizer_type: "SGD"
  learning_rate: 0.0001
  weight_decay: 0.01
  momentum: 0.9
  verbose: False
  scheduler_type: "CosineAnnealingLR"

# Model
model:
  type: "resnet" #
  ch_in: 12
  num_classes: 2
  dir_ckpt: "/data03/gkim/stem_cell_jwshin/outs/24_SEC1H5_patch_v4_allh_GMGL13_keep1_resnetp34bam_bin_lossbal"
  fname_ckpt: "epoch[00079]_tr[0.949]_va[0.936]_te[0.931]_trW[0.954]_vaW[0.945]_teW[0.936].pth.tar"
  aug_arch:
    ch_ft: 128
    growth_rate: 2
    block: "Bottleneck"
    layers: [3,4,6,3]
    att_type: "BAM"
    gap: False

# Dataset
data:
  dir_data: "/workspace03/gkim/stem_cell_jwshin/data/24_c0512_p0512_SEC1H5_wider_v4_testICCall_RS/"
  dir_infer: "/data03/gkim/stem_cell_jwshin/outs/24_SEC1H5_patch_v4_allh_GMGL13_keep1_resnetp34bam_bin_lossbal/infer"
  size_xy: 512
  size_z: 12
  reset_class: False

```
gpus: the id of GPU devices to use to compute deep neural networks and related gradients  
cpus: the number of CPU workers for torch DataLoader objects in the execution.  

loss: string patterns that indicate which type of classification loss to utilize  
batch_size: number of patch data to perform each inference or gradient calculation   
epoch: number of iterations over the entire dataset, which the training proceeds until  
optimizer_type: type of gradient-based neural network parameter optimizer  
scheduler_type: type of learning rate schedulers for the optimizer  

ch_in: number of input channels (number of used vertical stacks in our case)  
dir_ckpt: the directory where neural network checkpoints, including the network parameters, will be saved  
fname_ckpt: the name of checkpoint file to load, when inferring or resuming training  
ch_ft: number of feature map channels after the initial convolution  
growth_rate: ratio of feature map increase between neighboring ResNet stages  
block: ResNet block structure - "Bottleneck"   
layers: a list containing the block counts of four ResNet stages  
att_type: type of attention mechanism - "BAM", "CBAM", "SA" available & not applied if not specified  
gap: boolean for using global average pooling   


dir_data: the directory that contains train, val, or test data (each file is a 3D HT patch)   
dir_infer: the directory to save the inference result  
size_xy: horizontal patch size (number of pixels)  
size_z: vertical patch size (number of pixels)  
reset_class: boolean for resetting the class annotation (True required for inference)  
  
## Run the code for training

```shell

python3 train.py --config "path to your config file"

```

## Run the code for testing

```shell

python3 infer.py --config "path to your config file"

```

## License
This project is open-sourced under the MIT license.
