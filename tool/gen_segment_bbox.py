import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import circle, line_aa, polygon, polygon_perimeter
import json
from skimage import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import skimage.measure, skimage.transform
import sys

LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

PARTS_NAME = ['head', 'Larm', 'Rarm', 'belly', 'Lleg', 'Rleg', 'hip']

PARTS_SEL = [[0, 1, 14, 15, 16, 17], [5, 6, 7], [2, 3, 4], [1, 2, 5, 8, 11], [11, 12, 13], [8, 9, 10], [8, 11]]

PARTS_BOUND = [[5, 10], [5, 10], [5, 10], [5, 5], [5, 5], [5, 5], [10, 20]]

def get_head_wh(pose_joints, img_size):
	component_count = 0
	save_componets = []
	for component in part:
		if pose_joints[component][0] == MISSING_VALUE or pose_joints[component][1] == MISSING_VALUE:
			continue
		else:
			component_count += 1
			save_componets.append(pose_joints[component])
	if component_count >= 2:
		x_cords = []
		y_cords = []
		for component in save_componets:
			x_cords.append(component[1])
			y_cords.append(component[0])
		xmin = min(x_cords)
		xmax = max(x_cords)
		ymin = min(y_cords)
		ymax = max(y_cords)
		final_bb.append([max(xmin - PARTS_BOUND[id][0], 1), max(ymin - PARTS_BOUND[id][1], 1), \
						min(xmax + PARTS_BOUND[id][0], img_size[0]), min(ymax + PARTS_BOUND[id][1], img_size[1])])
	else:
		final_bb.append([0, 0, 0, 0])

def cal_bounding_boxes(pose_joints, img_size):
	final_bb = []
	id = 0
	for part in PARTS_SEL:
		component_count = 0
		save_componets = []
		for component in part:
			if pose_joints[component][0] == MISSING_VALUE or pose_joints[component][1] == MISSING_VALUE:
				continue
			else:
				component_count += 1
				save_componets.append(pose_joints[component])
		if component_count >= 2:
			x_cords = []
			y_cords = []
			for component in save_componets:
				x_cords.append(component[1])
				y_cords.append(component[0])
			xmin = min(x_cords)
			xmax = max(x_cords)
			ymin = min(y_cords)
			ymax = max(y_cords)
			final_bb.append([max(xmin - PARTS_BOUND[id][0], 1), max(ymin - PARTS_BOUND[id][1], 1), \
							min(xmax + PARTS_BOUND[id][0], img_size[0]), min(ymax + PARTS_BOUND[id][1], img_size[1])])
		else:
			final_bb.append([0, 0, 0, 0])
		id += 1
	print(final_bb)
	return final_bb

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

if __name__ == "__main__":
    import pandas as pd
    from skimage.io import imread, imsave
    import pylab as plt
    import os

    if not os.path.exists('./fashion_data/trainB/'):
    	os.mkdir('./fashion_data/trainB/')

    if not os.path.exists('./fashion_data/testB/'):
    	os.mkdir('./fashion_data/testB/')
    	
    # for train split
    df = pd.read_csv('./fashion_data/fasion-resize-annotation-train.csv', sep=':')
    for index, row in df.iterrows():
        print(row['name'])
        pose_cords = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
        final_cords = cal_bounding_boxes(pose_cords, (176, 256))
        np.save(os.path.join('./fashion_data/trainB/', row['name']), np.array(final_cords, dtype=np.float))

    df = pd.read_csv('./fashion_data/fasion-resize-annotation-test.csv', sep=':')
    for index, row in df.iterrows():
        print(row['name'])
        pose_cords = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
        final_cords = cal_bounding_boxes(pose_cords, (176, 256))
        np.save(os.path.join('./fashion_data/testB/', row['name']), np.array(final_cords, dtype=np.float))
