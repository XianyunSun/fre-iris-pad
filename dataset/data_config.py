import os
import random
from numpy import equal
import pandas as pd

import pdb

'''
pad_label=0 for attack samples, 
pad_label=1 for live samples
phase: 'train', 'val' or 'all'
SLA=True if consider soft lenses as attack samples
'''

def get_equal_data(image_list, label_list):
    image_live, image_att = [], []
    for i in range(len(image_list)):
        if label_list[i]==0: image_att.append(image_list[i])
        elif label_list[i]==1: image_live.append(image_list[i])
    
    if len(image_att)<len(image_live):
        image_live = random.sample(image_live, len(image_att))

    elif len(image_live)<len(image_att):
        image_att = random.sample(image_att, len(image_live))

    image_equal = image_live+image_att
    label_equal = [1 for i in range(len(image_live))] + [0 for j in range(len(image_att))]
    return image_equal, label_equal


class LivDet2023_Config(object):
    # train - live samples: 21188, # attack samples: 21919
    # val - live samples: 7062, # attack samples: 7304
    # test - live samples: 7062, # attack samples: 7305
    def __init__(self, img_type='raw', phase='all', fold='1', pai=False):
        self.img_type = img_type
        if img_type=='norm':
            self.root_path = None
        elif img_type=='roi':
            self.root_path = r'/rdata/xianyun.sun/iris_pad/Data_org_roi'
        elif img_type=='raw':
            self.root_path = r'/rdata/xianyun.sun/iris_pad/Data_org'

        self.metadata_train = r'/rdata/xianyun.sun/iris_pad/Data_org/FM_train'+fold+'_pai.csv'
        self.metadata_val = r'/rdata/xianyun.sun/iris_pad/Data_org/FM_val'+fold+'_pai.csv'
        self.metadata_test = r'/rdata/xianyun.sun/iris_pad/Data_org/FM_test'+fold+'_pai.csv'
        self.data_name = 'LivDet2023_' + phase + '_fold' + fold
        self.phase = phase
        self.pai = pai
    
    def read_data_phase(self, phase):
        df = pd.read_csv(getattr(self, 'metadata_'+phase))
        if self.img_type=='roi':
            image_list = [os.path.join(self.root_path, (df['image'][i]+'.png')[1:]) for i in range(len(df['image']))]
        else:
            image_list = [os.path.join(self.root_path, (df['image'][i]+df['format'][i])[1:]) for i in range(len(df['image']))]
        
        if self.pai:
            label_list = list(df['pad_type'])
        else:
            label_list = [1 if i=='bonafide' else 0 for i in list(df['label'])]
        return image_list, label_list

    def read_data(self, equal=False, SLA=False):
        img_train, label_train = self.read_data_phase('train')
        img_val, label_val = self.read_data_phase('val')
        img_test, label_test = self.read_data_phase('test')

        if self.phase=='train' or self.phase=='val' or self.phase=='test':
            image_list, label_list = locals()['img_'+self.phase], locals()['label_'+self.phase]
        elif self.phase=='train+val':
            image_list = img_train + img_val
            label_list = label_train + label_val
        elif self.phase=='all':
            image_list = img_train + img_val + img_test
            label_list = label_train + label_val + label_test
        elif self.phase=='val+test':
            image_list = img_val+img_test
            label_list = label_val+label_test
        
        if equal:
            image_list, label_list = get_equal_data(image_list, label_list)
        
        print('SLA setting not available')
        if self.pai:
            print('using pai mode! type of bonafide is 0')
            print('dataset: %s, # live samples: %d, # attack samples: %d.' 
                % (self.data_name, label_list.count(0), len(label_list)-label_list.count(0)))
        else:
            print('dataset: %s, # live samples: %d, # attack samples: %d.' 
                    % (self.data_name, label_list.count(1), label_list.count(0)))
        
        return image_list, label_list

    def get_name(self):
        return self.data_name

class LivDet2023_test_Config(object):
    # # live samples: 6500, # attack samples: 6832
    def __init__(self, img_type='raw'):
        self.img_type = img_type
        if img_type=='norm':
            self.root_path = None
        elif img_type=='roi':
            self.root_path = r'/rdata/xianyun.sun/iris_pad/LivDet2023_test_roi/images/'
        elif img_type=='raw':
            self.root_path = r'/rdata/xianyun.sun/iris_pad/LivDet2023_test/imagse/'

        self.metadata = r'/rdata/xianyun.sun/iris_pad/LivDet2023_test/livdet_by_pai.csv'
        self.data_name = 'LivDet2023_test'

    def read_data(self, SLA=False, equal=False):
        df = pd.read_csv(self.metadata)
        image_list = [os.path.join(self.root_path, df['filename'][i]) for i in range(len(df['filename']))]
        label_list = [1 if l=='Live'else 0 for l in df['ground_truth']]

        if equal:
            image_list, label_list = get_equal_data(image_list, label_list)
        
        print('SLA setting not available')
        print('dataset: %s, # live samples: %d, # attack samples: %d.' 
                % (self.data_name, label_list.count(1), label_list.count(0)))
        
        return image_list, label_list
    
    def get_name(self):
        return self.data_name

if __name__=='__main__':
    data = LivDet2023_test_Config('raw', phase='train')
    img_list, label = data.read_data()
    print(img_list[0])
    print(label[0])
    print(len(label))
