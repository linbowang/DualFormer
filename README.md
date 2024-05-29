# Label-aware Attention Network with Multi-scale Boosting for Medical Image Segmentation
This is the official implementation for: **Label-aware Attention Network with Multi-scale Boosting for Medical Image Segmentation**

## Preparation

#### 1. Datasets

  + Download the dataset from:

    Google Drive: https://drive.google.com/file/d/1TWeNqCt6Vo3ChTXjHg6wWqiIjxp_oVi8/view?usp=drive_link

    Baidu Netdisk：https://pan.baidu.com/s/1mn0VZujzAccNlYqyTJbk5g?pwd=ja0n 
    
  + Dataset is ordered as follow:

```
|-- data
|   |-- GlaS
|   |   |-- train
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- test
|   |   |   |-- images
|   |   |   |-- masks
|   |-- TNBC
|   |   |-- train
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- test
|   |   |   |-- images
|   |   |   |-- masks
|   |-- MoNuSeg
|   |   |-- train
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- test
|   |   |   |-- images
|   |   |   |-- masks
|   |-- CryoNuSeg
|   |   |-- train
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- test
|   |   |   |-- images
|   |   |   |-- masks
|   |-- PanNuke
|   |   |-- fold1
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- fold2
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- fold3
|   |   |   |-- images
|   |   |   |-- masks
```

#### 2.Pretrained models

  + Download the **ResNeXt101** pretrained model weights and put it in **‘./pretrained_model’**. https://drive.google.com/file/d/1R-sr-D6tdS3ZRoZCmJRl3IQrlDN59Lhp/view?usp=drive_link
  + Download the **pvt_v2_b2** pretrained model weights and put it in **‘./pretrained_model’**. https://drive.google.com/file/d/1n4EYUPz5HzGQeH6QbVm-eeQx8akQ5pMl/view?usp=drive_link

## Training

+ Run **train_PanNuke.py** to train model on the **PanNuke** dataset. 
+ Run **train_cross_validation.py** to train model on **GlaS**, **TNBC**, **MoNuSeg** and **CryoNuSeg**.

+ Trained models will be saved at **'./model/[dataset name]'**

## Environment

The code is developed on one NVIDIA RTX 4090 GPU with 24 GB memory and tested in Python 3.7.

pytorch  1.8.0
torchaudio  0.8.0
torchvision  0.13.1
numpy  1.21.5
timm  0.5.4
scipy  1.7.3
scikit-learn  0.24.2

