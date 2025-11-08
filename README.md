# Physically Realizable Adversarial Creating Attack  Against Vision-Based BEV Space  3D Object Detection

This is a official code release of [BEVDet-Spoofing](https://ieeexplore.ieee.org/abstract/document/10838314)（Physically Realizable Adversarial Creating Attack  Against Vision-Based BEV Space  3D Object Detection）. This code is mainly based on MMDetection3D（v1.0.0rc4）(https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc4/mmdet3d/models)、[BEVDet](https://github.com/HuangJunJie2017/BEVDet).



# Model Zoo

# Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. After data preparation, you will be able to see the following directory structure like(as is indicated in mmdetection3d):

notice: use our tools/create_data_bevdet.py

```
BEVDet-spoofing
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── bevdetv2-nuscenes_infos_train.pkl
│   │   ├── bevdetv2-nuscenes_infos_val.pkl

```

# Getting Started

```
conda create -n bevdetspoofing python=3.8
conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch
pip install openmim
mim install mmcv-full==1.5.3
mim install mmdet==2.25.1
mim install mmsegmentation==0.25.0
pip install pycuda \
    lyft_dataset_sdk \
    networkx==2.2 \
    numba==0.53.0 \
    numpy \
    nuscenes-devkit \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh==2.35.39 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install cumm-cu113
pip install spconv-cu113
cd BEVDet-spoofing
python setup.py develop
```

### Training

```
cd BEVDet-spoofing
python ./Spoofing3D/Train_Spoofing.py ./configs/bevdet/LidarSys-bevdet-r50-cbgs-spatial_0.6.py ./checkpoints/LidarSys_bevdet_r50_cbgs_spatial_06_mAP_3174_NDS_3939.pth
```

### Test

```
python ./Spoofing3D/eval.py ./configs/bevdet/LidarSys-bevdet-r50-cbgs-spatial_0.6.py ./checkpoints/LidarSys_bevdet_r50_cbgs_spatial_06_mAP_3174_NDS_3939.pth --poster_dir Spoofing3D/work_dir/try8/poster_30.pth
```

### Inference

```
python ./Spoofing/inference.py ./configs/bevdet/LidarSys-bevdet-r50-cbgs-spatial_0.6.py ./checkpoints/LidarSys_bevdet_r50_cbgs_spatial_06_mAP_3174_NDS_3939.pth --poster_dir Spoofing3D/work_dir/try8/poster_30.pth --id 1500
```



# Dependency

Our released implementation is tested on.

- Ubuntu  20.04
- Python 3.8
- NVIDIA CUDA 11.1
- A40 GPU

