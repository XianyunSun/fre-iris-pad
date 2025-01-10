'''
The code is based on paper: CBAM: Convolutional Block Attention Module
The official github is: https://github.com/Jongchan/attention-module
'''

import torch
from torch import nn

import numpy as np
import torch.nn.init as init
import numpy as np
import types
import torch.nn.functional as F

import pdb

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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], norm='sigmoid'):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.norm = norm
        assert norm=='sigmoid' or norm=='softmax'
    
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        if self.norm=='sigmoid':
            scale = torch.sigmoid(channel_att_sum)
        else:
            scale = F.softmax(channel_att_sum, dim=-1)
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class SpatialGate(nn.Module):
    def __init__(self, kernel_size, norm='sigmoid', seperate='none'): # seperate='none', 'seperate1', 'seperate2'
        super(SpatialGate, self).__init__()
        self.kernel_size = kernel_size #3
        self.norm = norm
        assert norm=='sigmoid' or norm=='softmax'

        self.seperate = seperate
        if seperate=='seperate1':
            self.spatial = BasicConv(4, 1, self.kernel_size , stride=1, padding=(self.kernel_size -1) // 2, relu=False)
        elif seperate=='seperate2':
            self.spatial = BasicConv(4, 2, self.kernel_size , stride=1, padding=(self.kernel_size -1) // 2, relu=False)
        else:
            self.spatial = BasicConv(2, 1, self.kernel_size , stride=1, padding=(self.kernel_size -1) // 2, relu=False)

    def forward(self, x):
        C = int(x.shape[1]/2.)
        if self.seperate=='seperate1' or self.seperate=='seperate2':
            x_freq = torch.cat( (torch.max(x[:, :C, :, :],1)[0].unsqueeze(1), torch.mean(x[:, :C, :, :],1).unsqueeze(1)), dim=1 )
            x_spa = torch.cat( (torch.max(x[:, C:, :, :],1)[0].unsqueeze(1), torch.mean(x[:, C:, :, :],1).unsqueeze(1)), dim=1 )
            x_compress = torch.cat((x_freq, x_spa), dim=1)
            x_attn = self.spatial(x_compress)
        else:
            x_compress = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 ) # channel pool
            x_attn = self.spatial(x_compress)
        
        if self.norm=='sigmoid':
            scale = torch.sigmoid(x_attn) # broadcasting
        else:
            b, c, h, w = x_attn.shape
            scale = F.softmax(x_attn.view(b, c, -1), dim=-1).view(b, c, h, w)

        if self.seperate=='seperate2':
            x_out_freq = x[:, :C, :, :] * scale[:, 0, :, :].unsqueeze(1)
            x_out_spa = x[:, C:, :, :] * scale[:, 1, :, :].unsqueeze(1)
            x_out = torch.cat((x_out_freq, x_out_spa), dim=1)
        else:
            x_out = x*scale

        return x_out

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class SpatialChannelGate(nn.Module):
    def __init__(self, gate_channels, kernel_size, norm='sigmoid', cat='para', seperate='none'):
        super(SpatialChannelGate, self).__init__()
        self.spatial = SpatialGate(kernel_size=kernel_size, norm=norm, seperate=seperate)
        self.channel = ChannelGate(gate_channels=gate_channels, norm=norm)
        self.cat = cat
        assert cat=='para' or cat=='cat'

    def forward(self, x):
        if self.cat=='para':
            x_spatial = self.spatial(x)
            x_channel = self.channel(x)
            x_out = 0.5*x_channel + 0.5*x_spatial
        elif self.cat=='cat':
            x_channel = self.channel(x)
            x_out = self.spatial(x_channel)
        return x_out


if __name__=='__main__':
    x = torch.randn(4, 640, 7, 7)
    model = SpatialChannelGate(gate_channels=64, kernel_size=7, seperate='seperate2')
    y = model(x)
    print(x.shape)