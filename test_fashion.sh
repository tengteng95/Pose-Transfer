#!/bin/bash

python test.py --dataroot ./fashion_data/ \
    --name prefashion_PATN \
    --model PATN \
    --phase test \
    --dataset_mode keypoint \
    --norm instance \
    --batchSize 1 \
    --resize_or_crop no \
    --gpu_ids 0 \
    --BP_input_nc 18 \
    --no_flip \
    --which_model_netG PATN \
    --checkpoints_dir ./checkpoints \
    --pairLst ./fashion_data/fasion-resize-pairs-test.csv \
    --which_epoch latest \
    --results_dir ./results \
    --display_id 0 \
