# Pose-Transfer
Code for the paper **Progressive Pose Attention for Person Image Generation** in **CVPR19(Oral)**. The paper is available [here](http://arxiv.org/abs/1904.03349). 

<p float="center">
	<img src='imgs/women1.jpg' width="100"/>
  	<img src='imgs/walkfront.gif' width="100"/>
  	<img src='imgs/women2.jpg' width="100"/>
	<img src='imgs/dance.gif' width="100"/>
	<img src='imgs/women3.jpg' width="100"/>
    <img src='imgs/dance2.gif' width="100"/>
    <img src='imgs/women4.jpg' width="100"/>
    <img src='imgs/dance3.gif' width="100"/>
</p>


## Requirement
* pytorch(0.3.1)
* torchvision(0.2.0)
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
We provide our **dataset split files** and **extracted keypoints files** for convience.


#### DeepFashion
- Download [deep fasion dataset in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). You will need to ask a pasword from dataset maintainers. 
- Split the raw images into the train split (```fashion_data/train```) and the test split (```fashion_data/test```). Launch
```bash
python tool/generate_fashion_datasets.py
``` 
- Download train/test pairs and train/test key points annotations from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg), including **fasion-resize-pairs-train.csv**, **fasion-resize-pairs-test.csv**, **fasion-resize-annotation-train.csv**, **fasion-resize-annotation-train.csv**. Put these four files under the ```fashion_data``` directory.
- Generate the pose heatmaps. Launch
```bash
python tool/generate_pose_map_fashion.py
```
- Generate body segements. Launch
```bash
python tool/gen_segment_bbox.py
```

#### Notes:
**Optionally, you can also generate these files by yourself.**

1. Keypoints files

We use [OpenPose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) to generate keypoints. 

- Download pose estimator from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg). Put it under the root folder ``Pose-Transfer``.
- Change the paths **input_folder**  and **output_path** in ``tool/compute_coordinates.py``. And then launch
```bash
python2 compute_coordinates.py
```

2. Dataset split files

```bash
python2 tool/create_pairs_dataset.py
```

<!-- #### Pose Estimation
- Download the pose estimator from [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).
- Launch ```python compute_cordinates.py``` to get the pose estimation for both datasets.

OR you can download our generated pose estimations from here. (Coming soon.) --> 

### Train a model

DeepFashion
```bash
python train.py --dataroot ./fashion_data/ --name fashion_PATN_Deform --model PATN_Deform --lambda_GAN 5 --lambda_A 1 --lambda_B 1 --lambda_style 10 --dataset_mode key_segments --n_layers 3 --norm instance --batchSize 4 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN_Deform --niter 500 --niter_decay 200 --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-train.csv --L1_type l1_plus_seperate_segments_style --which_model_netD resnet_in --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 0 --no_lsgan
```


### Test the model

DeepFashion
```bash
python test.py --dataroot ./fashion_data/ --name fashion_PATN_Deform --model PATN_Deform --phase test --dataset_mode key_segments --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN_Deform --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch latest --results_dir ./results
```

### Evaluation
We adopt SSIM, mask-SSIM, IS, mask-IS, DS, and PCKh for evaluation of Market-1501. SSIM, IS, DS, PCKh for DeepFashion.

#### 1) SSIM and mask-SSIM, IS and mask-IS, mask-SSIM

For evaluation, **Tensorflow 1.4.1(python3)** is required. Please see ``requirements_tf.txt`` for details.

For Market-1501:
```bash
python tool/getMetrics_market.py
```

For DeepFashion:
```bash
python tool/getMetrics_market.py
```

If you still have problems for evaluation, please consider using **docker**. 

```bash
docker run -v <Pose-Transfer path>:/tmp -w /tmp --runtime=nvidia -it --rm tensorflow/tensorflow:1.4.1-gpu-py3 bash
# now in docker:
$ pip install scikit-image tqdm 
$ python tool/getMetrics_market.py
```

Refer to [this Issue](https://github.com/tengteng95/Pose-Transfer/issues/4).

#### 2) DS Score
Download pretrained on VOC 300x300 model and install propper caffe version [SSD](https://github.com/weiliu89/caffe/tree/ssd). Put it in the ssd_score forlder. 

For Market-1501:
```bash
python compute_ssd_score_market.py --input_dir path/to/generated/images
```

For DeepFashion:
```bash
python compute_ssd_score_fashion.py --input_dir path/to/generated/images
```

#### 3) PCKh

- First, run ``tool/crop_market.py`` or ``tool/crop_fashion.py``.
- Download pose estimator from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg). Put it under the root folder ``Pose-Transfer``.
- Change the paths **input_folder**  and **output_path** in ``tool/compute_coordinates.py``. And then launch
```bash
python2 compute_coordinates.py
```
- run ``tool/calPCKH_fashion.py`` or ``tool/calPCKH_market.py``



### Pre-trained model 
Our pre-trained model can be downloaded [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg).


## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{zhu2019progressive,
  title={Progressive Pose Attention Transfer for Person Image Generation},
  author={Zhu, Zhen and Huang, Tengteng and Shi, Baoguang and Yu, Miao and Wang, Bofei and Bai, Xiang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2347--2356},
  year={2019}
}
```

### Acknowledgments
Our code is based on the popular [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).