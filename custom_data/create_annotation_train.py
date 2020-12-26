from torchvision.models.detection import keypointrcnn_resnet50_fpn as krcnn
import torch
import numpy as np
from PIL import Image
import os
from glob import glob
import sys


device = torch.device("cuda:0")
model = krcnn(pretrained=True).to(device)
model.eval()


def reorder(points_torch, scores_torch):
    points_patn = [np.array([-1, -1, 1]) for _ in range(18)]
    scores_patn = [0 for _ in range(18)]
    neck = np.array([
        [(points_torch[5][0] + points_torch[6][0]) // 2, (points_torch[5][1] + points_torch[6][1]) // 2, 1]
    ])
    neck_score = np.array([(scores_torch[5] + scores_torch[6]) / 2])
    scores_torch = np.concatenate([scores_torch, neck_score], axis=0)
    points_torch = list(np.concatenate([points_torch, neck], axis=0))
    patn_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    for i, o in enumerate(patn_order):
        points_patn[i] = points_torch[o]
        scores_patn[i] = scores_torch[o]
    return np.array(points_patn), np.array(scores_patn)


def get_keypoints(fpath):
    with torch.no_grad():
        base_path = os.path.basename(fpath)
        image = np.array(Image.open(os.path.join("train", base_path)))
        im = image.transpose(2, 0, 1) / 255
        ten = torch.from_numpy(im).unsqueeze(0).float().to(device)
        result = model(ten)[0]
        idx = np.argmax(result["scores"].cpu().detach().numpy())
        points = result["keypoints"].cpu().detach().numpy()[idx]
        point_scores = result["keypoints_scores"].cpu().detach().numpy()[idx]

        points, point_scores = reorder(points, point_scores)
        xs, ys = [], []
        for i, point in enumerate(points):
            if point_scores[i] > 5:
                xs.append(int(point[0]))
                ys.append(int(point[1]))
            else:
                xs.append(-1)
                ys.append(-1)
        return ":".join([base_path, str(ys), str(xs)])


ann_file = open("fasion-resize-annotation-train.csv", 'w')
pair_file = open("fasion-resize-pairs-train.csv", 'w')
ann_file.write("name:keypoints_y:keypoints_x\n")
pair_file.write("from,to\n")

fpaths = glob(sys.argv[1])
for i, fpath in enumerate(fpaths):
    base_path = os.path.basename(fpath)
    Image.open(fpath).resize((176, 256)).save(os.path.join("train", base_path))
    ann_file.write(get_keypoints(fpath) + "\n")
    for j, fpath_to in enumerate(fpaths):
        if i != j:
            base_path_to = os.path.basename(fpath_to)
            pair_file.write(f"{base_path},{base_path_to}\n")

ann_file.close()
pair_file.close()
