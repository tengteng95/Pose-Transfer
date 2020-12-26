#!/bin/bash

source activate pytorch_p36
python ../../emoticon_data/refine_images.py . .
python create_annotation_test.py '*.png'
python generate_pose_map_custom.py test
