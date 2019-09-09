import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.SegmentsStyleLoss import SegmentsSeperateStyleLoss

import sys
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.phase = opt.phase
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_BBox1_set = self.Tensor(nb, opt.nsegments, 4)
        self.input_BBox2_set = self.Tensor(nb, opt.nsegments, 4)
        self.num_of_Goutput = 1

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        # num. of down-sampling blocks
                                        n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            # for resD, dropout is selective.
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            # for resD, dropout is selective.
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = SegmentsSeperateStyleLoss(opt.nsegments, opt.lambda_A, opt.lambda_B, opt.lambda_style, opt.perceptual_layers, self.gpu_ids[0])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_PB = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netD_PB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netD_PP.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']

        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)

        input_BBox1, input_BBox2 = input['BBox1'], input['BBox2']
        self.input_BBox1_set.resize_(input_BBox1.size()).copy_(input_BBox1)
        self.input_BBox2_set.resize_(input_BBox2.size()).copy_(input_BBox2)

        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]


    def forward(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        self.input_BBox1 = Variable(self.input_BBox1_set)
        self.input_BBox2 = Variable(self.input_BBox2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1), 
                   self.input_BBox1, 
                   self.input_BBox2]

        self.fake_p2 = self.netG(G_input)

    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)
        
        self.input_BBox1 = Variable(self.input_BBox1_set)
        self.input_BBox2 = Variable(self.input_BBox2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1), 
                   self.input_BBox1, 
                   self.input_BBox2]
        
        self.fake_p2 = self.netG(G_input)


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_unpaired_G(self):

        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN


        # L1 loss
        l1_losses = self.criterionL1(self.fake_p2, self.input_P2, self.input_BBox2_set)
        self.loss_l1_all = l1_losses[0]
        self.loss_originL1 = l1_losses[1]
        self.loss_perceptual = l1_losses[2]
        self.loss_style = l1_losses[3]


        pair_L1loss = self.loss_l1_all 
        pair_loss = pair_L1loss + pair_GANloss

        pair_loss.backward()

        self.pair_L1loss = pair_L1loss.data[0]
        self.pair_GANloss = pair_GANloss.data[0]


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
        self.loss_D_PB = loss_D_PB.data[0]

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)
        self.loss_D_PP = loss_D_PP.data[0]


    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_unpaired_G()
        self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()

        # D_BP
        for i in range(self.opt.DG_ratio):
            self.optimizer_D_PB.zero_grad()
            self.backward_D_PB()
            self.optimizer_D_PB.step()


    def get_current_errors(self):
        # ('pair_L1loss', self.pair_L1loss),
        ret_errors = OrderedDict([('D_PB', self.loss_D_PB), 
                                 ('pair_L1loss', self.pair_L1loss),
                                 ('pair_GANloss', self.pair_GANloss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP

        ret_errors['origin_L1'] = self.loss_originL1
        ret_errors['perceptual'] = self.loss_perceptual
        ret_errors['style'] = self.loss_style

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        fake_p2 = util.tensor2im(self.fake_p2.data)
        

        vis = np.zeros((height, width*5, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_P2
        vis[:, width*3:width*4, :] = input_BP2
        vis[:, width*4:, :] = fake_p2


        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)

        self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

