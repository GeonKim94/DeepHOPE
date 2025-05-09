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
  cpus: 5

# Optimizer
optim:
  loss: "CE"
  batch_size: 64
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
  aug_arch:
    ch_ft: 128
    growth_rate: 2
    block: "Bottleneck"
    layers: [3,4,6,3]
    att_type: "BAM"
    gap: False
    
  
  dir_ckpt: "/data03/gkim/stem_cell_jwshin/outs/24_SEC1H5_patch_v4_allh_GMGL13_keep1_resnetp34bam_bin_lossbal"
  fname_ckpt: "epoch[00079]_tr[0.949]_va[0.936]_te[0.931]_trW[0.954]_vaW[0.945]_teW[0.936].pth.tar"

# Dataset
data:
  dir_data: "/workspace03/gkim/stem_cell_jwshin/data/24_c0512_p0512_SEC1H5_wider_v4_testICCall_RS/"
  dir_infer: "/data03/gkim/stem_cell_jwshin/outs/24_SEC1H5_patch_v4_allh_GMGL13_keep1_resnetp34bam_bin_lossbal/infer"
  size_xy: 512
  size_z: 12
  reset_class: False


```


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
