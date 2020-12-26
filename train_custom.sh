#!/bin/bash
source activate pytorch_031
cd custom_data
./make_dataset_train.sh
cd ..

python train.py --dataroot ./custom_data/ \
    --name custom_PATN \
    --model PATN \
    --lambda_GAN 5 \
    --lambda_A 1 \
    --lambda_B 1 \
    --dataset_mode keypoint \
    --n_layers 3 \
    --norm instance \
    --batchSize 7 \
    --pool_size 0 \
    --resize_or_crop no \
    --gpu_ids 0 \
    --BP_input_nc 18 \
    --no_flip \
    --which_model_netG PATN \
    --niter 50 \
    --niter_decay 20 \
    --checkpoints_dir ./checkpoints \
    --pairLst ./custom_data/fasion-resize-pairs-train.csv \
    --L1_type l1_plus_perL1 \
    --n_layers_D 3 \
    --with_D_PP 1 \
    --with_D_PB 1  \
    --display_id 0
