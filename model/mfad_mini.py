import torch
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

import numpy as np
import torch.nn.functional as F
import timm
import pdb

import sys
sys.path.append('../')
from model.attention import ChannelGate, SpatialGate, SpatialChannelGate


class FAD_HAM_Net_mini_FFT_SpCh2(nn.Module):
    def __init__(self, pad_classes=2, image_shape=(3, 224, 224), pretrain=None, variant='resnet50', attn=True, fft='fft', norm='sigmoid', input='aug', cat='para', seperate='none'):
        super(FAD_HAM_Net_mini_FFT_SpCh2, self).__init__()
        self.fft = fft
        self.input = input
        assert input=='aug' or input=='ori'
        assert cat=='para' or cat=='cat'
        print('using {} for frequency transformation' .format(fft))
        print('processing {} image' .format(input))
        self.FAD_Head = FAD_Head(image_shape[-1], channel=['l', 'm'])
        if fft=='dct':
            size = image_shape[-1]
            self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
            self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        if 'resnet' in variant:
            self.FFT_encoder = ResNetEncoder(in_channels=2 if fft=='fft' else 1, variant=variant)
            self.AUG_encoder = ResNetEncoder(in_channels=image_shape[0], variant=variant)

            if pretrain is not None:
                state_dict = torch.load(pretrain)
                load_matched_state_dict(self.FFT_encoder, state_dict, print_stats=True)
                load_matched_state_dict(self.AUG_encoder, state_dict, print_stats=True)
        else:
            self.FFT_encoder = TimmEncoder(variant, ckpt=pretrain, input_dim=2 if fft=='fft' else 1)
            self.AUG_encoder = TimmEncoder(variant, ckpt=pretrain, input_dim=image_shape[0])

        # get backbone output shape
        tmp = torch.randn(2, image_shape[0], image_shape[1], image_shape[2])
        with torch.no_grad():
            tmp_fad, tmp_fad3, tmp_fad2, tmp_fad1 = self.AUG_encoder(tmp)

        self.spch1 = SpatialChannelGate(kernel_size=5, gate_channels=tmp_fad1.shape[1]*2, norm=norm, cat=cat, seperate=seperate)
        self.spch3 = SpatialChannelGate(kernel_size=7, gate_channels=tmp_fad3.shape[1]*2, norm=norm, cat=cat, seperate=seperate)

        if attn:
            self.low_filter_weight = nn.Sequential(
                nn.Conv2d(2, 4, kernel_size=3, padding='same'),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.Conv2d(4, 1, kernel_size=3, padding='same')
            )
            print('using attention in mfad_head model')
        else: self.low_filter_weight = None

        self.downsample_to3 = nn.Upsample(size=(tmp_fad3.shape[2], tmp_fad3.shape[3]), mode='bilinear', align_corners=False)
        self.downsample_to1 = nn.Upsample(size=(tmp_fad1.shape[2], tmp_fad1.shape[3]), mode='bilinear', align_corners=False)

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.lastconv = nn.Sequential(
            nn.Conv2d((tmp_fad3.shape[1]+tmp_fad1.shape[1])*2, tmp_fad2.shape[1]*2, 
                        kernel_size=1, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(tmp_fad2.shape[1]*2),)
        
        # for binary supervision
        self.linear1 = nn.Linear(tmp_fad2.shape[1]*2, tmp_fad2.shape[1], bias=False)
        self.bn = nn.BatchNorm1d(tmp_fad2.shape[1])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.6)
        self.cls = nn.Linear(tmp_fad2.shape[1], pad_classes, bias=False)

    def forward(self, x, x_aug):
        # attn
        low_filter = self.FAD_Head(x[:, 0, ...].unsqueeze(1))
        if self.low_filter_weight is not None:    
            attn = torch.sigmoid(self.low_filter_weight(low_filter))
        else: 
            attn = torch.ones(size=(low_filter.shape[0], 1, low_filter.shape[2], low_filter.shape[3])).to(low_filter.device)

        # fft
        if self.input=='ori':
            x_aug = x   # using original image
        
        if self.fft=='fft':
            fftr = torch.fft.fft2(x_aug[:, 0, ...].unsqueeze(1))
            ffta = torch.roll(torch.abs(fftr), shifts=(fftr.size(-2)//2, fftr.size(-1)//2), dims=(-2, -1))
            fftp = torch.roll(torch.angle(fftr), shifts=(fftr.size(-2)//2, fftr.size(-1)//2), dims=(-2, -1))
            fre = torch.cat((ffta, fftp), dim=1)
        elif self.fft=='dct':
            fre = self._DCT_all @ x_aug @ self._DCT_all_T
            fre = fre[:, 0, ...].unsqueeze(1)
        
        fad, fad_3, fad_2, fad_1 = self.FFT_encoder(fre)
        aug, aug_3, aug_2, aug_1 = self.AUG_encoder(x_aug)
        concate_3 = torch.cat((fad_3, self.downsample_to3(attn)*aug_3), dim=1)
        concate_1 = torch.cat((fad_1, self.downsample_to1(attn)*aug_1), dim=1)
        # aug_1:40*48*48, aug_2:112*24*24, aug_3:320*12*12

        # high-level feature map using channel attention
        att_x_3 = self.spch3(concate_3) # 14x14, 3
        att_x_3_14x14 = self.downsample_to3(att_x_3)

        att_x_1 = self.spch1(concate_1) # 56x56, 7
        att_x_1_14x14 = self.downsample_to3(att_x_1)

        x_concate = torch.cat((att_x_1_14x14, att_x_3_14x14), dim=1)    # B*720*12*12

        pad_feats = self.pooling(x_concate) # B*720*1*1
        pad_feats = self.lastconv(pad_feats).squeeze() # B*224

        # MLP classification head
        x = self.linear1(pad_feats) # B*112
        #x = self.bn(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.cls(x)

        # output feats shape: 384
        return {'pad':x, 'pad_feats':pad_feats}


'''
Timm encoder for using other timm models as backbones
'''
class TimmEncoder(nn.Module):
    def __init__(self, model_name, ckpt=None, input_dim=3):
        super(TimmEncoder, self).__init__()
        if ckpt is not None:
            backbone = timm.create_model(model_name, num_classes=0, pretrained=True, pretrained_cfg_overlay=dict(file=ckpt))
        else: backbone = timm.create_model(model_name, num_classes=0, pretrained=False)
        backbone = nn.ModuleList(backbone.children())
        if 'efficientnet' in model_name:
            backbone[0] = nn.Conv2d(input_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.conv = nn.Sequential(*backbone[0:2])
            blocks = nn.ModuleList(backbone[2].children())
            self.block1 = nn.Sequential(*blocks[0:3])
            self.block2 = nn.Sequential(*blocks[3:5])
            self.block3 = nn.Sequential(*blocks[5:])
            self.head = nn.Sequential(*backbone[3:6])
        elif 'densenet' in model_name:
            blocks = nn.ModuleList(backbone[0].children())
            blocks[0] = nn.Conv2d(input_dim,  64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.conv = nn.Sequential(*blocks[0:3])
            self.block1 = nn.Sequential(*blocks[3:7])
            self.block2 = nn.Sequential(*blocks[7:9])
            self.block3 = nn.Sequential(*blocks[9:])
            self.head = nn.Sequential(*backbone[1:])

    def forward(self, img):
        img = self.conv(img)
        img1 = self.block1(img)
        img2 = self.block2(img1)
        img3 = self.block3(img2)
        img = self.head(img3)
        return img, img3, img2, img1

'''
FAD_Head and Filter is based on paper: Thinking in frequency: Face forgery detection by mining frequency-aware clues
The corresponding code is based on an unofficial implementation: https://github.com/yyk-wew/F3Net
'''
class FAD_Head(nn.Module):
    def __init__(self, size, channel=['h', 'all'], learnable=True):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1 || 0 - 1
        low_filter = Filter(size, 0, size // 16, use_learnable=learnable)
        middle_filter = Filter(size, size // 16, size // 8, use_learnable=learnable)
        high_filter = Filter(size, size // 8, size, use_learnable=learnable)
        all_filter = Filter(size, 0, size, use_learnable=learnable)

        channel_list = {'l':low_filter, 'm':middle_filter, 'h':high_filter, 'all':all_filter}
        self.filters = nn.ModuleList([channel_list[name] for name in channel])
        #self.filters = nn.ModuleList([middle_filter, high_filter, all_filter]) # mh
        #self.filters = nn.ModuleList([high_filter, all_filter]) #h
        

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 224, 224]

        # 4 kernel
        y_list = []
        for i in range(len(self.filters)):
            x_pass = self.filters[i](x_freq)  # [N, 3, 224, 224]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 224, 224]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 224, 224]
        return out

# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

# ResNet is used as backbone
class ResNetEncoder(ResNet):
    #[3, 4, 6, 3],
    layers = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
    }
    blocks = {
        'resnet18': BasicBlock,
        'resnet34': BasicBlock,
        'resnet50': Bottleneck,
        'resnet101': Bottleneck,        
    }

    def __init__(self, in_channels=3, variant='resnet50', norm_layer=None):
        super().__init__(
            block=self.blocks[variant],
            layers=self.layers[variant],
            replace_stride_with_dilation=[False, False, False],
            norm_layer=norm_layer)

        expansion = 4

        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        # Delete fully-connected layer
        del self.fc

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56  
        x1 = x

        x = self.layer1(x) # 56
        x = self.layer2(x) # 28
        x2 = x

        x = self.layer3(x)
        x = self.layer4(x) #14
        x3 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x, x3, x2, x1

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=False, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


if __name__ == "__main__":
    
    image_x = torch.randn(4, 3, 384, 384)
    fad_ham = FAD_HAM_Net_mini_FFT_SpCh2(variant='efficientnet_b0', pretrain=None, image_shape=(3, 384, 384), norm='sigmoid', input='aug', cat='para', seperate='seperate1', fft='dct', bk=False)
    pdb.set_trace()

    y = fad_ham(image_x, image_x, train=False)
    print('binary output shape:', y['pad'].shape, 'feature map shape:', y['pad_feats'].shape)
    #print(y['mask_map'].shape)