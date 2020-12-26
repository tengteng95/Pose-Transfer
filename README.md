# Pose-Transfer
This repository is forked [this](https://github.com/tengteng95/Pose-Transfer) repo.

This repo is modified for the purpose of the enhancement of the model upon different domains of dataset.

## Requirement
The model performs best when met with below requirements

* pytorch(0.3.1)
* torchvision(0.2.0)
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate

You can install pytorch 0.3.1 by
```{commandline}
$ conda install pytorch=0.4.1 cuda90 -c pytorch
```
Rest of the package can be installed in an orthodox method with pip.

## Data Preperation
Preparation of the original dataset can be found at the original repo.

### Preparing custom data
Move all of your image files in custom_data/image_set

### Train
```{commandline}
$ ./train_custom.sh
```

### Test
```{commandline}
$ ./test_custom.sh
```

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
