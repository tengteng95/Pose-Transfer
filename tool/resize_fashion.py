from skimage.io import imread, imsave
from skimage.transform import resize
import os
import numpy as np
import pandas as pd
import json

def resize_dataset(folder, new_folder, new_size = (256, 176), crop_bord=40):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for name in os.listdir(folder):
        old_name = os.path.join(folder, name)
        new_name = os.path.join(new_folder, name)

        img = imread(old_name)
        if crop_bord == 0:
            pass
        else:
            img = img[:, crop_bord:-crop_bord]

        img = resize(img, new_size, preserve_range=True).astype(np.uint8)

        imsave(new_name, img)

def resize_annotations(name, new_name, new_size = (256, 176), old_size = (256, 256), crop_bord=40):
    df = pd.read_csv(name, sep=':')

    ratio_y = new_size[0] / float(old_size[0])
    ratio_x = new_size[1] / float(old_size[1] - 2 * crop_bord)

    def modify(values, ratio, crop):
        val = np.array(json.loads(values))
        mask = val == -1
        val = ((val - crop) * ratio).astype(int)
        val[mask] = -1
        return str(list(val))

    df['keypoints_y'] = df.apply(lambda row: modify(row['keypoints_y'], ratio_y, 0), axis=1)
    df['keypoints_x'] = df.apply(lambda row: modify(row['keypoints_x'], ratio_x, crop_bord), axis=1)

    df.to_csv(new_name, sep=':', index=False)


root_dir = 'xxx'
resize_dataset(root_dir + '/test', root_dir + 'fashion_resize/test')
resize_annotations(root_dir + 'fasion-annotation-test.csv', root_dir + 'fasion-resize-annotation-test.csv')

resize_dataset(root_dir + '/train', root_dir + 'fashion_resize/train')
resize_annotations(root_dir + 'fasion-annotation-train.csv', root_dir + 'fasion-resize-annotation-train.csv')


