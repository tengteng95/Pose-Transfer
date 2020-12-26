#!/bin/bash

source activate pytorch_p36
python ../../emoticon_data/refine_images.py image_set image_set
python create_annotation_train.py 'image_set/*.png'
python generate_pose_map_custom.py train custom
