# Pytorch StereoNet
Customized implementation of the [Stereonet guided hierarchical refinement for real-time edge-aware depth prediction](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper.pdf)

****************************************************************
I have just noticed that the share link above is not available now. Please download from this link: https://drive.google.com/file/d/1CO2hGMphCZsNcJPFAk2q4-QG_0zM0LpU/view?usp=sharing.

This project is the first one after I have learned Pytorch. And it's the homework for my CV course. I'm focusing on the segmentation field at present. Thanks for your attention.
****************************************************************

![The network archtecture of StereoNet](https://raw.githubusercontent.com/zhixuanli/StereoNet/master/stereo-net-structure.png)

## Attention: Not accomplished yet
1. Stereo Matching is not my main research field, and this repo is created for a homework. So maybe it's not very completed, but I have tried to make it perfect. If you need a better version, please refer to [https://github.com/meteorshowers/StereoNet](https://github.com/meteorshowers/StereoNet).
2. The approach of computing the cost volume in the StereoNet paper is subtracting the padding image and the other image. Here I changed it to concatenate the two images. If you want to change it to the paper's way, just set it when you initialize the net.
3. Only training and testing on the KITTI 2015 train dataset is not enough, the best performance has achieved 74.5% (pixels with error smaller than 1). After pretraining on SceneFlow and finetune on KITTI15, the acc achieves 90.054%, not as good as the acc in paper. I have try hard to achieve the accuracy in paper, but still can't. Maybe some details are wrong. 
 

### Experiment Results till now
1. train and test on SceneFlow datasets:
    + epoch 22 total training loss = 3.956
    + average test EPE = 3.496
2. different finetuning on kitti 15 and result
    + 300 epochs, max 3 pixel error rate = 80.893 on kitti val 
        ```
        optimizer = RMSprop(model.parameters(), lr=1e-3, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        ```
    + 300 epochs, max 3 pixel error rate = 83.527 on kitti val 
        ```
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        ```
    + 300 epochs, max 3 pixel error rate = 90.054 on kitti val 
        ```
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if epoch <= 200:
            lr = 0.001
        else:
            lr = 0.0001    
        ```
    + 2000 epochs, max 3 pixel error rate = 93.680 on kitti val, after 4.98 hours finetune 
        ```
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        if epoch <= 200:
            lr = 0.001
        else:
            lr = 0.0001    
        ```

## Pre-requirement
+ Pytorch 1.0.0
+ CUDA Toolkit 10
+ numpy

### Datasets:
1. Pretrain: [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
2. [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

### You can use the anaconda virtual environment to quick start

#### Install Anaconda
```
1. wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
2. bash Anaconda3-5.3.1-Linux-x86_64.sh
```

Please reference to [Ubuntu系统下Anaconda使用方法总结](https://www.cnblogs.com/QingHuan/p/9987069.html) for more information about conda installation.

#### Create Virtual Environment according to my environment index 
```
conda env create -n your_env_name -f environment.yaml
```

## Training and Test
#### Switch to the correct python environment
```
conda activate your_env_name
```

#### Start training and test
Pretrain on SceneFlow dataset
```
cd pretrain-sceneflow
python sceneflow-pretrain.py
```

Finetune on KITTI 2015
```
cd finetune-kitti15
python finetune-kitti15.py
```

## Coding Reference
+ [Pyramid Stereo Matching Network (CVPR2018)](https://github.com/JiaRenChang/PSMNet)
+ [gc-net for stereo matching by using pytorch](https://github.com/zyf12389/GC-Net)
+ [An tensorflow raw implementation of paper "End-to-End Learning of Geometry and Context for Deep Stereo Regression"](https://github.com/liuruijin17/GCNet-tensorflow)
