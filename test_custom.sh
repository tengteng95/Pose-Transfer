#!/bin/bash
source activate pytorch_031

python test.py --dataroot ./custom_data/ \
    --name custom_PATN \
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
    --pairLst ./custom_data/fasion-resize-pairs-train.csv \
    --which_epoch latest \
    --results_dir ./results \
    --display_id 0 \
