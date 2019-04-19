from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

class L1_plus_perceptualLoss(nn.Module):
    def __init__(self, lambda_L1, lambda_perceptual, perceptual_layers, gpu_ids, percep_is_l1):
        super(L1_plus_perceptualLoss, self).__init__()

        self.lambda_L1 = lambda_L1
        self.lambda_perceptual = lambda_perceptual
        self.gpu_ids = gpu_ids

        self.percep_is_l1 = percep_is_l1

        vgg = models.vgg19(pretrained=True).features
        self.vgg_submodel = nn.Sequential()
        for i,layer in enumerate(list(vgg)):
            self.vgg_submodel.add_module(str(i),layer)
            if i == perceptual_layers:
                break
        self.vgg_submodel = torch.nn.DataParallel(self.vgg_submodel, device_ids=gpu_ids).cuda()

        print(self.vgg_submodel)

    def forward(self, inputs, targets):
        if self.lambda_L1 == 0 and self.lambda_perceptual == 0:
            return Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)), Variable(torch.zeros(1))
        # normal L1
        loss_l1 = F.l1_loss(inputs, targets) * self.lambda_L1

        # perceptual L1
        mean = torch.FloatTensor(3)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = Variable(mean)
        mean = mean.resize(1, 3, 1, 1).cuda()

        std = torch.FloatTensor(3)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = Variable(std)
        std = std.resize(1, 3, 1, 1).cuda()

        fake_p2_norm = (inputs + 1)/2 # [-1, 1] => [0, 1]
        fake_p2_norm = (fake_p2_norm - mean)/std

        input_p2_norm = (targets + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm = (input_p2_norm - mean)/std


        fake_p2_norm = self.vgg_submodel(fake_p2_norm)
        input_p2_norm = self.vgg_submodel(input_p2_norm)
        input_p2_norm_no_grad = input_p2_norm.detach()

        if self.percep_is_l1 == 1:
            # use l1 for perceptual loss
            loss_perceptual = F.l1_loss(fake_p2_norm, input_p2_norm_no_grad) * self.lambda_perceptual
        else:
            # use l2 for perceptual loss
            loss_perceptual = F.mse_loss(fake_p2_norm, input_p2_norm_no_grad) * self.lambda_perceptual

        loss = loss_l1 + loss_perceptual

        return loss, loss_l1, loss_perceptual

