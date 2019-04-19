# Market1501
python test.py --dataroot ./market_data/ --name market_PATN_test --model PATN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 2 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --pairLst ./market_data/market-pairs-test.csv --which_epoch latest --results_dir ./results

# DeepFashion
python test.py --dataroot ./fashion_data/ --name fashion_PATN_test --model PATN --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch latest --results_dir ./results
