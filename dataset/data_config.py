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
