import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.feature import local_binary_pattern
import random
import copy
from collections import Counter
from scipy.fftpack import dct
#import torchshow as ts
#import torch_dct as dct
#import math

import sys
sys.path.append('../')
from dataset.data_config import *

import pdb

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]

def get_sample_weights_combined(dataset_list):
    sample_weight_combined = []
    for dataset in dataset_list:
        class_counts = dataset._dataframe.label.value_counts()
        sample_weights = [1 / class_counts[i] for i in dataset._dataframe.label.values]
        sample_weight_combined.extend(sample_weights)

    return sample_weight_combined

def build_transform():
    transform_uni = A.ReplayCompose([
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),
            A.RandomGamma(gamma_limit=(80, 180)), # 0.5, 1.5
            #A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            A.OneOf([
                A.GaussianBlur(p=.2),
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
            #A.GaussNoise(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=45, p=0.5),
            # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            A.RandomBrightnessContrast(p=0.5), 
        ])
    return transform_uni

class UniformSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_class, label='domain'):
        self.data_source = data_source
        self.num_domain = num_class

        self.indices_domain = [[] for _ in range(num_class)]
        for i, data in enumerate(self.data_source):
            self.indices_domain[data[label]].append(i)
        
    def __iter__(self):
        count = 0
        current_class = 0
        indices = copy.deepcopy(self.indices_domain)
        random.shuffle(indices)

        while count < len(self.data_source):
            if len(indices[current_class]) > 0:
                yield indices[current_class].pop()
                count += 1
            else: break
            current_class = (current_class + 1) % self.num_domain


    def __len__(self):
        return len(self.data_source)

    def __len__(self):
        return len(self.data_source)

class BasicDataset(Dataset):
    def __init__(self, config_list, img_size=(224,224), equal=False, aug=[], SLA=False, train=False, split=-1., indexs=None):
        super(BasicDataset, self).__init__()
        self.image_list = []
        self.label_list = []
        self.domain_label_list = []
        self.triplet_label_list = [] # 0 for live and 1,2,3... for different fake domains
        self.aug = aug

        for i, domain in enumerate(config_list):
            image_list, label_list = domain.read_data(equal=equal, SLA=SLA)
            self.image_list += image_list
            self.label_list += label_list
            self.domain_label_list += [i for j in range(len(image_list))]
            self.triplet_label_list += [0 if label_list[k]==1 else i+1 for k in range(len(label_list))]

        # train_test_split
        if split>0 and indexs is None:
            indexs = list(range(len(self.image_list)))
            self.train_index, self.test_index = train_test_split(indexs, test_size=split, random_state=42)
            self.image_list = [self.image_list[m] for m in self.train_index]
            self.label_list = [self.label_list[m] for m in self.train_index]
            self.domain_label_list = [self.domain_label_list[m] for m in self.train_index]
            self.triplet_label_list = [self.triplet_label_list[m] for m in self.train_index]
        elif indexs is not None:
            self.image_list = [self.image_list[m] for m in indexs]
            self.label_list = [self.label_list[m] for m in indexs]
            self.domain_label_list = [self.domain_label_list[m] for m in indexs]
            self.triplet_label_list = [self.triplet_label_list[m] for m in indexs]

        if train:
            self.transform_train = build_transform()
        else: self.transform_train = None
        self.transform_totensor = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),])

    def get_weight_sampler(self):
        class_dict = Counter(self.label_list)
        weight_list = [1./class_dict[pad] for pad in self.label_list]
        sampler = WeightedRandomSampler(weights=weight_list, num_samples=len(self.image_list), replacement=True)
        return sampler

    def __getitem__(self, index):
        output = {}
        img = cv2.imread(self.image_list[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print(self.image_list[index])

        if self.transform_train is not None:
            replay = self.transform_train(image=img)
            img_ori = replay['image']
        else: 
            replay = None
            img_ori = img

        output['img'] = self.transform_totensor(image=img_ori)['image']

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if 'CLAHE' in self.aug:
            img_clahe = np.expand_dims(img, -1)
            cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = cla.apply(img_clahe)
            img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
            output['CLAHE'] = self.transform_totensor(image=img_clahe)['image']
        
        if 'LBP' in self.aug:
            radius = 1 
            n_points = 8 * radius
            img_name = self.image_list[index].split('/')[-1]
            img_format = img_name.split('.')[-1]
            img_lbp_name = img_name.replace('.'+img_format, '_LBP_r'+str(radius)+'_n'+str(n_points)+'.'+img_format)
            img_lbp_path = self.image_list[index].replace(img_name, img_lbp_name)
            if os.path.exists(img_lbp_path):
                img_lbp = cv2.imread(img_lbp_path)
                img_lbp = cv2.cvtColor(img_lbp, cv2.COLOR_BGR2RGB)
                if replay is not None:
                    img_lbp = A.ReplayCompose.replay(replay['replay'], image=img_lbp)['image']
            else:
                #print('process LBP')
                img_lbp = local_binary_pattern(img, n_points, radius)
                img_lbp = cv2.cvtColor(img_lbp.astype('uint8'), cv2.COLOR_GRAY2RGB)
            output['LBP'] = self.transform_totensor(image=img_lbp)['image']
            
        output['pad'] = self.label_list[index]
        output['domain'] = self.domain_label_list[index]
        output['triplet'] = self.triplet_label_list[index]
        #print(self.image_list[index], label, domain_label, triplet_label)
        
        return output
    
    def __len__(self):
        return len(self.image_list)


if __name__=='__main__':
    #config_list = [NDCLD_Config('roi'), NDIris3D_AD100_Config('roi'), NDIris3D_LG4000_Config('roi')]
    config_list = [LivDet2023_test_Config('roi')]
    #config_list = [NDCLD13_AD100_Config('raw')]
    data = BasicDataset(config_list, aug=['LBP'], train=True, split=0.2)

    #sampler = UniformSampler(data, num_class=2, label='pad')

    pdb.set_trace()
    dataloader = torch.utils.data.DataLoader(data, batch_size=2, sampler=None, drop_last=True, shuffle=True)
    print('len dataloader:', len(dataloader))







