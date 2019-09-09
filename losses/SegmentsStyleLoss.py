from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from roi_align.roi_align import RoIAlign

def to_varabile(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        # print(b, c, h, w)
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(c * h * w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), GramMatrix()(target))
        return (out)


class SegmentsSeperateStyleLoss(nn.Module):
    # PARTS_NAME = ['head', 'Larm', 'Rarm', 'belly', 'Lleg', 'Rleg', 'hip']
    def __init__(self, nsegments, lambda_L1, lambda_perceptual, lambda_style, perceptual_layers, gpu_ids, roi_output_size=
                [[16, 16], [16, 7], [16, 7], [32, 32], [32, 7], [32, 7], [16, 16]]):
        super(SegmentsSeperateStyleLoss, self).__init__()
        self.nsegments = nsegments
        self.align_layer_lists = nn.ModuleList([RoIAlign(x[0], x[1], transform_fpcoor=True) for x in roi_output_size])
        # print(self.align_layer_lists)
        self.lambda_L1 = lambda_L1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_style = lambda_style
        self.gpu_ids = gpu_ids

        vgg = models.vgg19(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg_submodel = nn.Sequential()
        for i,layer in enumerate(list(vgg)):
            self.vgg_submodel.add_module(str(i),layer)
            if i == perceptual_layers:
                break
        self.vgg_submodel = self.vgg_submodel.cuda(gpu_ids)
        # print('####perceptual sub model defination!####')
        # print(self.vgg_submodel)

    def deal_rois(self, rois):
        batchSize = rois.shape[0]
        indexes = [[] for i in range(self.nsegments)]
        results = [[] for i in range(self.nsegments)]
        for i in range(batchSize):
            for seg in range(self.nsegments):
                bbox = rois[i, seg, :].cpu().int().numpy()
                if bbox[0] == bbox[1] == bbox[2] == bbox[3] == 0:
                    continue
                indexes[seg].append(i)
                results[seg].append(bbox)
        return indexes, results

    def forward(self, inputs, targets, rois):
        box_index_data, boxes_data = self.deal_rois(rois)

        # normal L1
        loss_l1 = F.l1_loss(inputs, targets) * self.lambda_L1

        # perceptual L1
        mean = torch.FloatTensor(3)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = Variable(mean)
        mean = mean.resize(1, 3, 1, 1)
        mean = mean.cuda(self.gpu_ids)

        std = torch.FloatTensor(3)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = Variable(std)
        std = std.resize(1, 3, 1, 1)
        std = std.cuda(self.gpu_ids)

        fake_p2_norm = (inputs + 1)/2 # [-1, 1] => [0, 1]
        fake_p2_norm = (fake_p2_norm - mean)/std

        input_p2_norm = (targets + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm = (input_p2_norm - mean)/std

        fake_p2_norm_perceptual = self.vgg_submodel(fake_p2_norm)
        input_p2_norm_perceptual = self.vgg_submodel(input_p2_norm)

        input_p2_norm_perceptual_no_grad = input_p2_norm_perceptual.detach()

        for i in range(self.nsegments):
            # perceptual loss
            boxes_array = np.asarray(boxes_data[i],dtype=np.float32)
            if boxes_array.shape[0] == 0:
                if i == 0:
                    loss_style = 0
                else:
                    loss_style += 0
                continue
            boxes = to_varabile(np.asarray(boxes_data[i],dtype=np.float32), requires_grad=False, is_cuda=True)
            box_index = to_varabile(np.asarray(box_index_data[i],dtype=np.int32), requires_grad=False, is_cuda=True)

            fake_perceptual_segments = self.align_layer_lists[i](fake_p2_norm_perceptual, boxes, box_index)
            input_perceptual_segments_no_grad = self.align_layer_lists[i](input_p2_norm_perceptual_no_grad, boxes, box_index)
            if i == 0:
                loss_style = GramMSELoss()(fake_perceptual_segments, input_perceptual_segments_no_grad) * self.lambda_style
            else:
                loss_style += GramMSELoss()(fake_perceptual_segments, input_perceptual_segments_no_grad) * self.lambda_style

        # input_perceptual_segments_no_grad = input_perceptual_segments.detach()

        loss_perceptual = F.l1_loss(fake_p2_norm_perceptual, input_p2_norm_perceptual_no_grad) * self.lambda_perceptual
        loss = loss_l1 + loss_style + loss_perceptual

        return loss, loss_l1, loss_perceptual, loss_style



