'''
Author: Alexsandr_Lim >> linbosen@stu.ouc.edu.cn
Date: 2023-11-17 13:31:19
LastEditors: Alexsandr_Lim >> linbosen@stu.ouc.edu.cn
LastEditTime: 2024-04-07 20:26:42
FilePath: /bosen/workspace/Underwater_IQA/my_NR-IQM/models/uiqa/mpvit_y_cbcr_res_dam.py
Description: 

Copyright (c) 2023 by Alexsandr_Lim, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from functools import partial
from einops import rearrange
from torch import nn, einsum

from kornia.color.ycbcr import rgb_to_ycbcr

from models.uiqa.res_block import BottleneckBlock, BasicBlock, ResBlock, CBAM_ResBlock
import math

def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32
    
    left_pad = (w_out- w) // 2
    right_pad = w_out - w - left_pad
    top_pad  = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad
    
    img = nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)
    
    return img

class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """
    """ Reference https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L49."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    """ Reference https://github.com/youngwanLEE/MPViT/blob/main/mpvit.py#L82"""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn_weight_init=1, norm_layer=nn.BatchNorm2d, act_layer=None,
                ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x

class DilatedConv2d_BN(nn.Module):
    """Convolution with BN module."""
    """ Reference https://github.com/youngwanLEE/MPViT/blob/main/mpvit.py#L82"""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn_weight_init=1, norm_layer=nn.BatchNorm2d, act_layer=None,
                ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation, groups=out_ch, bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x
    
class DWConv2d_BN(nn.Module):
    """Depthwise Separable Convolution with BN module."""
    """ Reference https://github.com/youngwanLEE/MPViT/blob/main/mpvit.py#L128"""
    def __init__(self, in_ch, out_ch, kernel_size=1, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.Hardswish, bn_weight_init=1,
                 ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation, groups=out_ch, bias=False)
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        """ foward function """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x
    
class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch Embedding."""
    """Reference: https://github.com/youngwanLEE/MPViT/blob/main/mpvit.py#L179"""
    def __init__(self, in_chans=3, embed_dim=768, patch_size=[3,3,3], dilation=[1,2,4], num_path=4, isPool=False,
                 act_layer=nn.Hardswish):
        super(DWCPatchEmbed, self).__init__()


        self.multi_scale_patch_embeds = nn.ModuleList([
            DilatedConv2d_BN(in_chans, embed_dim, kernel_size=patch_size[idx], dilation=dilation[idx],
                        stride=2 if (isPool and idx==0) else 1, 
                        act_layer=act_layer
            ) for idx in range(num_path)
        ])
        
    def forward(self, x):
        """foward function"""
        out = []
        for pe in self.multi_scale_patch_embeds:
            x = pe(x)
            out.append(x)
        return out


class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding.
        Note: This module is similar to the conditional position encoding in CPVT. 
        Reference https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L161
    """
    def __init__(self, dim, k=3):
        """init function"""
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x

class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding.
                    It can have two forms:
                    1. An integer of window size, which assigns all attention headswith the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size
                                        2: #attention head split 2})
                    It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W).contiguous()
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h).contiguous()

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat

class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    """ Reference https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L119 """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)                                           # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v).contiguous()          # Shape: [B, h, Ch, Ch].
        factor_att        = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v).contiguous()  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)                                                # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)                                        # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x                                                                         # Shape: [B, N, C].

class MHC_Attention(nn.Module):
    """Multi-Head Convolutional self-Attention block."""
    """ Similar tp SerialBlock in CoAT """
    """ Reference https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L188"""

    def __init__(self, dim, num_heads, mlp_ratio=3, qkv_bias=True, qk_scale=None, attn_drop=0., drop=0.,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 shared_cpe=None, shared_crpe=None,
    ):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x, size):
        """foward function"""
        # Conv-Attention.
        if self.cpe is not None:        # Apply convolutional position encoding.
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))  # Apply factorized attention and convolutional relative position encoding.

        # MLP. 
        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x

class MHC_Attention_Encoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA` blocks."""
    def __init__(self, dim, blk_depth=1, num_heads=8, mlp_ratio=3, qkv_bias=True, qk_scale=None, attn_drop=0., drop=0.,
                 drop_path_list=[], act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),  crpe_window={3:2, 5:3, 7:3}):
        super().__init__()

        self.blk_depth = blk_depth
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        
        self.MHCA_layers = nn.ModuleList([
            MHC_Attention(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop,
                          drop_path=drop_path_list[idx], act_layer=act_layer, norm_layer=norm_layer,
                          shared_cpe=self.cpe, shared_crpe=self.crpe,
                          ) for idx in range(self.blk_depth)
        ])

    def forward(self, x, size):
        """foward function"""
        H, W = size
        B = x.shape[0]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
class Res_Encoder(nn.Module):
    """ Residual encoder block for convolutional local feature."""
    """ https://github.com/youngwanLEE/MPViT/blob/main/mpvit.py#L471"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Hardswish, norm_layer=nn.BatchNorm2d):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(in_features, hidden_features, kernel_size=1, act_layer=act_layer)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, 
                                bias=False, groups=hidden_features)
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features, kernel_size=1,)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ initialization"""
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat
    

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_path=4, blk_depth=1, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 attn_drop=0., drop=0., drop_path_list=[]):
        super().__init__()

        self.transformer_encoders = nn.ModuleList([
            MHC_Attention_Encoder(dim=in_dim, blk_depth=blk_depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, drop=drop, 
                                  drop_path_list=drop_path_list,
            )for _ in range(num_path)
        ])

        self.conv_encoder = Res_Encoder(in_features=in_dim, out_features=in_dim)
        self.feature_interaction = Conv2d_BN(in_ch=in_dim*(num_path+3), out_ch=out_dim, act_layer=nn.Hardswish)
    
    def forward(self, inputs, y_inputs, cbcr_inputs):
        """foward function"""
        attention_outputs = []

        # Convolitional Local Feature.
        attention_outputs.append(self.conv_encoder(inputs[0]))
        # Transformer Encoder.
        for x, encoder in zip(inputs, self.transformer_encoders):
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()    # [B, C, H, W] -> [B, N, C]
            attention_outputs.append(encoder(x, size=(H, W)))

        attention_outputs.append(y_inputs)
        attention_outputs.append(cbcr_inputs)
        # Global-to-Local Feature Interaction.
        out_concat = torch.cat(attention_outputs, dim=1)
        out = self.feature_interaction(out_concat)

        return out

class Cls_Head(nn.Module):
    """a linear layer for classification."""
    def __init__(self, embed_dim, num_classes):
        """initialization"""
        super().__init__()
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """foward function"""
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1).contiguous()          # (B, C, H, W) -> (B, C, 1)
        out = self.cls(x)           # [B, C]
        return out
    

class Ycbcr_Block(nn.Module):
    """ YCbCr Block for YCbCr feature extraction."""
    """ Reference"""
    def __init__(self, in_dim, out_dim, nb_layer=1, bootleneck=True, isPool=False):
        super().__init__()
        
        y_in_dim = in_dim
        cbcr_in_dim = in_dim
        y_out_dim = out_dim
        cbcr_out_dim = out_dim

        self.isPool = isPool

        self.y_pool = nn.MaxPool2d(2)
        self.y_block = CBAM_ResBlock(nb_layer, y_in_dim, y_out_dim, block=BottleneckBlock if bootleneck else BasicBlock, 
                                     use_cam=False, use_sam=False)


        self.cbcr_pool = nn.MaxPool2d(2)
        self.cbcr_block = CBAM_ResBlock(nb_layer, cbcr_in_dim, cbcr_out_dim, block=BottleneckBlock if bootleneck else BasicBlock, 
                                   use_cam=True, use_sam=True)


    
    def forward(self, x_y, x_cbcr):
        """foward function"""
        if self.isPool: 
            x_y = self.y_pool(x_y)
            x_cbcr = self.cbcr_pool(x_cbcr)

        y_out = self.y_block(x_y)
        cbcr_out = self.cbcr_block(x_cbcr)

        out = torch.cat([y_out, cbcr_out], dim=1)
        return out, y_out, cbcr_out

class MultiLayerCls_Head(nn.Module):
    """a linear layer for classification."""
    def __init__(self, embed_dim, num_classes):
        """initialization"""
        super().__init__()

        self.cls = nn.Sequential(
            # Linear + LReLu + Linear + LReLu + Linear
            nn.Linear(embed_dim, embed_dim // 4, bias=False),  # 2C -> 2C // 4
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim // 4, embed_dim // 16, bias=False),  # 2C // 4 -> 2C // 16
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim // 16, num_classes, bias=False),  # 2C // 16 -> 1
        )

    def forward(self, x):
        """foward function"""
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1).contiguous()          # (B, C, H, W) -> (B, C, 1)
        out = self.cls(x)           # [B, C]
        return out


class PAUQA(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 num_stages=4, num_paths=[4, 4, 4, 4], patch_size=[[3,3],[3,3,3],[3,3,3],[3,3,3]], dilation_size=[[1,2],[1,2,4],[1,2,4],[1,2,4]],
                 blk_depths=[1, 1, 1, 1], embed_dims=[64, 128, 256, 512], res_dims=[2, 2, 2, 2],
                 mlp_ratios=[8, 8, 4, 4], num_heads=[8, 8, 8, 8], qkv_bias=True, qk_scale=None, drop_path_rate=0.0):
    
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages

        dpr = self.dpr_generator(drop_path_rate, blk_depths, num_stages)

        self.ConvStem = nn.Sequential(
            Conv2d_BN(in_chans, embed_dims[0]//2, kernel_size=3, stride=2, padding=1, act_layer=nn.Hardswish),
            Conv2d_BN(embed_dims[0]//2, embed_dims[0], kernel_size=3, stride=2, padding=1, act_layer=nn.Hardswish),
        )

        # Patch embeddings.
        self.patch_embed1 = DWCPatchEmbed(in_chans=embed_dims[0], embed_dim=embed_dims[0], patch_size=patch_size[0], dilation=dilation_size[0],
                                          num_path=num_paths[0], isPool=False)
        self.patch_embed2 = DWCPatchEmbed(in_chans=embed_dims[1], embed_dim=embed_dims[1], patch_size=patch_size[1], dilation=dilation_size[1], 
                                          num_path=num_paths[1], isPool=True)
        self.patch_embed3 = DWCPatchEmbed(in_chans=embed_dims[2], embed_dim=embed_dims[2], patch_size=patch_size[2], dilation=dilation_size[2], 
                                          num_path=num_paths[2], isPool=True)
        self.patch_embed4 = DWCPatchEmbed(in_chans=embed_dims[3], embed_dim=embed_dims[3], patch_size=patch_size[3], dilation=dilation_size[3], 
                                          num_path=num_paths[3], isPool=True)

        # Class tokens.
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))


        # Blocks.
        self.blocks1 = Block(in_dim=embed_dims[0], out_dim=embed_dims[1], blk_depth=blk_depths[0], num_path=num_paths[0], 
                             num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, 
                             drop_path_list=dpr[0])
        
        self.blocks2 = Block(in_dim=embed_dims[1], out_dim=embed_dims[2], blk_depth=blk_depths[1], num_path=num_paths[1], 
                             num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, 
                             drop_path_list=dpr[1])
        
        self.blocks3 = Block(in_dim=embed_dims[2], out_dim=embed_dims[3], blk_depth=blk_depths[2], num_path=num_paths[2], 
                             num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, 
                             drop_path_list=dpr[2])
        
        self.blocks4 = Block(in_dim=embed_dims[3], out_dim=embed_dims[3], blk_depth=blk_depths[3], num_path=num_paths[3], 
                             num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, 
                             drop_path_list=dpr[3])

        # Ycbcr Blocks.
        y_init_dim = embed_dims[0]
        cbcr_init_dim = embed_dims[0]

        self.y_init_convs = nn.Sequential(
                nn.MaxPool2d(2),
                Conv2d_BN(in_chans//3, y_init_dim, kernel_size=3, stride=1, padding=1, act_layer=nn.ReLU),
                Conv2d_BN(y_init_dim, y_init_dim, kernel_size=3, stride=1, padding=1, act_layer=nn.ReLU),
                )
        self.cbcr_init_convs = nn.Sequential(
                nn.MaxPool2d(2),
                Conv2d_BN(2*in_chans//3, cbcr_init_dim, kernel_size=3, stride=1, padding=1, act_layer=nn.ReLU),
                Conv2d_BN(cbcr_init_dim, cbcr_init_dim, kernel_size=3, stride=1, padding=1, act_layer=nn.ReLU),
                )

        self.ycbcr_blocks1 = Ycbcr_Block(in_dim=embed_dims[0], out_dim=embed_dims[0], nb_layer=res_dims[0], isPool=True)
        self.ycbcr_blocks2 = Ycbcr_Block(in_dim=embed_dims[0], out_dim=embed_dims[1], nb_layer=res_dims[1], isPool=True)
        self.ycbcr_blocks3 = Ycbcr_Block(in_dim=embed_dims[1], out_dim=embed_dims[2], nb_layer=res_dims[2], isPool=True)
        self.ycbcr_blocks4 = Ycbcr_Block(in_dim=embed_dims[2], out_dim=embed_dims[3], nb_layer=res_dims[3], isPool=True)

        # Classification head(s).
        self.cls_head = Cls_Head(embed_dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def forward(self, x):
        x, x_ycbcr = self.preprocessing(x)

        x = self.forward_features(x, x_ycbcr)
        
        pred = self.cls_head(x)
            
        return pred


    
    def forward_features(self, x, x_ycbcr):
        BS, C, H, W = x.shape

        x = self.ConvStem(x)         # [bs*3*384*384] -> [bs*64*96*96]

        y0_y = self.y_init_convs(x_ycbcr[:,0:1,:,:].contiguous())     # [bs*1*384*384] -> [bs*64*96*96]
        y0_cbcr = self.cbcr_init_convs(x_ycbcr[:,1:,:,:].contiguous())    # [bs*2*384*384] -> [bs*128*96*96]
        y0 = torch.cat([y0_y, y0_cbcr], dim=1)

        # Serial blocks 1.              [bs*64*96*96] -> [bs*128*96*96]
        y1, y1_y, y1_cbcr = self.ycbcr_blocks1(y0_y, y0_cbcr)
        x1 = self.patch_embed1(x)
        x1 = self.blocks1(x1, y1_y, y1_cbcr)
        
        # Serial blocks 2.              [bs*128*96*96] -> [bs*216*48*48]
        y2, y2_y, y2_cbcr = self.ycbcr_blocks2(y1_y, y1_cbcr)
        x2 = self.patch_embed2(x1)
        x2 = self.blocks2(x2, y2_y, y2_cbcr)
        
        # Serial blocks 3.              [bs*216*48*48] -> [bs*288*24*24]
        y3, y3_y, y3_cbcr = self.ycbcr_blocks3(y2_y, y2_cbcr)
        x3 = self.patch_embed3(x2)
        x3 = self.blocks3(x3, y3_y, y3_cbcr)
        
        # Serial blocks 4.              [bs*288*24*24]-> [bs*288*12*12]
        y4, y4_y, y4_cbcr = self.ycbcr_blocks4(y3_y, y3_cbcr)
        x4 = self.patch_embed4(x3)
        x4 = self.blocks4(x4, y4_y, y4_cbcr)

        return x4

    def preprocessing(self, x):
        x = padding_img(x)
        x_ycbcr = rgb_to_ycbcr(x)
        return x, x_ycbcr

    def dpr_generator(self, drop_path_rate, num_layers, num_stages):
        """Generate drop path rate list following linear decay rule."""
        dpr_list = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]
        dpr = []
        cur = 0
        for i in range(num_stages):
            dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
            dpr.append(dpr_per_stage)
            cur += num_layers[i]

        return dpr

    def insert_cls_token(self, x, cls_token):
        """ Insert CLS token. """
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    
    def remove_cls_token(self, x):
        """ Remove CLS token. """
        return x[:, 1:, :].contiguous()


