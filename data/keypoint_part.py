import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase)
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'K')
        self.dir_BBox = os.path.join(opt.dataroot, opt.phase + 'B')

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        # person 1 and its bone

        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, P2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_B, P1_name + '.npy') # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name) # person 2
        BP2_path = os.path.join(self.dir_B, P2_name + '.npy') # bone of person 2

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path)
        BP2_img = np.load(BP2_path)

        BBox1_path = os.path.join(self.dir_BBox, P1_name + '.npy')
        BBox2_path = os.path.join(self.dir_BBox, P2_name + '.npy')
        BBox1 = np.load(BBox1_path)
        BBox2 = np.load(BBox2_path)

        BBox1 = torch.from_numpy(BBox1)
        BBox2 = torch.from_numpy(BBox2)

        BP1 = torch.from_numpy(BP1_img).float() #h, w, c
        BP1 = BP1.transpose(2, 0) #c,w,h
        BP1 = BP1.transpose(2, 1) #c,h,w 

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0) #c,w,h
        BP2 = BP2.transpose(2, 1) #c,h,w 

        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)
        
        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'BBox1': BBox1, 'BBox2': BBox2,
                'P1_path': P1_name, 'P2_path': P2_name}

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'
