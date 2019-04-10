# Pose-Transfer
Code for the paper **Progressive Pose Attention for Person Image Generation** in CVPR19. The paper is available [here](http://arxiv.org/abs/1904.03349).

<img src='imgs/results.png' width=800>

This is Pytorch implementation for pose transfer on both Market1501 and DeepFashion dataset. The code is written by [Tengteng Huang](https://github.com/tengteng95) and [Zhen Zhu](https://github.com/jessemelpolio).

## Requirement
* pytorch 0.3.1
* torchvision
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate


## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/tengteng95/Pose-Transfer.git
cd Pose-Transfer
```

### Data Preperation
#### Market1501
- Download the Market1501 dataset from [here](http://www.liangzheng.com.cn/Project/project_reid.html).

#### DeepFashion
- Download the DeepFashion dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

#### Pose Estimation
- Download the pose estimator from [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).
- Launch ```python compute_cordinates.py``` to get the pose estimation for both datasets.

OR you can download our generated pose estimations from here. (Coming soon.) 

### Train a model


### Test the model


### Pre-trained model 

## Citation
If you use this code for your research, please cite our paper.
```
```

### Acknowledgments
Our code is based on the popular [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).