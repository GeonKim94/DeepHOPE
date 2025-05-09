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

## Use appropriate path for the following parameters
In the main.py, 

``` shell

--data_dir = 'xxx/dataset'
--ckpt_dir = 'xxx/model-gastric/ckpt'
--result_dir = 'where you want to save the results'

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
