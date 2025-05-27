'''
Author: Alexsandr_Lim >> linbosen@stu.ouc.edu.cn
Date: 2024-03-17 22:38:09
LastEditors: Alexsandr_Lim >> linbosen@stu.ouc.edu.cn
LastEditTime: 2024-03-21 15:14:59
FilePath: /bosen/workspace/Underwater_IQA/my_NR-IQM/models/uiqa/dilated_conv_attention_blocks.py
Description: 

Copyright (c) 2024 by Alexsandr_Lim, All Rights Reserved. 
'''
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F


# https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/_deeplab.py

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, dilation=1):

        padding = dilation * (kernel - 1) // 2

        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    

class DilatedAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, atrous_kernels):
        super(DilatedAttentionModule, self).__init__()
        modules = []
        # modules.append(nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        kernel1, kernel2, kernel3 = tuple(atrous_kernels)
        modules.append(ASPPConv(in_channels, out_channels, kernel1, rate1))
        modules.append(ASPPConv(in_channels, out_channels, kernel2, rate2))
        modules.append(ASPPConv(in_channels, out_channels, kernel3, rate3))
        # modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            )
        

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)




