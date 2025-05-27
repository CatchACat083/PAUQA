from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.uiqa.dilated_conv_attention_blocks import DilatedAttentionModule

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Channel_Attention(nn.Module):
    '''Channel Attention in CBAM.
    '''

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        '''Param init and architecture building.
        '''

        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types

        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''

        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scaled #return the element-wise multiplication between the input and the result.


class ChannelPool(nn.Module):
    '''Merge all the channels in a feature map into two separate channels where the first channel is produced by taking the max values from all channels, while the
       second one is produced by taking the mean from every channel.
    '''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_Attention(nn.Module):
    '''Spatial Attention in CBAM.
    '''

    def __init__(self, kernel_size=7):
        '''Spatial Attention Architecture.
        '''

        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        scaled = nn.Sigmoid()(x_output)
        return x * scaled


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_cam: bool = True,
        use_sam: bool = True
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_cam: bool = True,
        use_sam: bool = True
    ) -> None:
        super().__init__()

        self.use_cam = use_cam
        self.use_sam = use_sam

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.use_cam:
            self.ca = Channel_Attention(planes * self.expansion)
        if self.use_sam:
            self.sa = Spatial_Attention()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if self.use_cam:
            out = self.ca(out)
        if self.use_sam:
            out = self.sa(out)

        out += identity
        out = self.relu(out)

        return out
    

class ResBlock(nn.Module):
    def __init__(self, nb_layer, in_planes, planes, block, groups=1, base_width=64, stride=1, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, planes, nb_layer, groups, base_width, stride, dilation, norm_layer)

    def _make_layer(self, block, in_planes, planes, nb_layer, groups=1, base_width=64, stride=1, dilation=1, norm_layer=None):
        downsample = None

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride),
                norm_layer(planes),
            )
        
        layers = []
        layers.append(block(in_planes, planes, stride, downsample, groups=groups, base_width=base_width, dilation=dilation))

        for _ in range(1, nb_layer):
            layers.append(block(planes, planes, groups=groups, base_width=base_width, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    

class DA_ResBlock(nn.Module):
    def __init__(self, nb_layer, in_planes, planes, block, groups=1, base_width=64, stride=1, dilation=1, norm_layer=None):
        super(DA_ResBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, planes, nb_layer, groups, base_width, stride, dilation, norm_layer)

    def _make_layer(self, block, in_planes, planes, nb_layer, groups=1, base_width=64, stride=1, dilation=1, norm_layer=None):
        downsample = None

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride),
                norm_layer(planes),
            )
        
        layers = []
        layers.append(DilatedAttentionModule(in_planes, in_planes, atrous_rates=[1, 2, 2], atrous_kernels=[1, 3, 5]))
        layers.append(block(in_planes, planes, stride, downsample, groups=groups, base_width=base_width, dilation=dilation))

        # for _ in range(1, nb_layer):
        #     layers.append(block(planes, planes, groups=groups, base_width=base_width, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    

class CBAM_ResBlock(nn.Module):
    def __init__(self, nb_layer, in_planes, planes, block, groups=1, base_width=64, stride=1, dilation=1, norm_layer=None,
                 use_cam=True, use_sam=True):
        super(CBAM_ResBlock, self).__init__()

        self.use_cam = use_cam
        self.use_sam = use_sam

        self.layer = self._make_layer(block, in_planes, planes, nb_layer, groups, base_width, stride, dilation, norm_layer)

    def _make_layer(self, block, in_planes, planes, nb_layer, groups=1, base_width=64, stride=1, dilation=1, norm_layer=None):
        downsample = None

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride),
                norm_layer(planes),
            )
        
        layers = []
        layers.append(block(in_planes, planes, stride, downsample, groups=groups, base_width=base_width, dilation=dilation,
                            use_cam=self.use_cam, use_sam=self.use_sam))

        for _ in range(1, nb_layer):
            layers.append(block(planes, planes, groups=groups, base_width=base_width, dilation=dilation,
                                use_cam=self.use_cam, use_sam=self.use_sam))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
    