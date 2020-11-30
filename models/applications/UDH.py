# _*_ coding: utf-8 _*_
# @Time : 2020/11/30 2:49 下午 
# @Author : Blue
# @File : UDH.py

import torch
import torch.nn as nn
from ..backbones.EfficientNet import EfficientNet
from ..necks.gap import GlobalAveragePooling
from ..heads.linear_head import LinearHead
from ..losses.photometric_loss import PhotometricLoss

class UDH(nn.Module):
    '''
    Implementation of Unsupervised Deep Homography
    '''
    def __init__(self, backbone_name, patch_size):
        super(UDH, self).__init__()
        self.backbone = EfficientNet.from_pretrained(backbone_name, in_channels=2, include_top=False)
        self.neck = GlobalAveragePooling()
        self.head = LinearHead(8, 1280)
        self.loss = PhotometricLoss(patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        patch_a = x['patch_a']
        patch_b = x['patch_b']
        x = torch.cat((patch_a, patch_b), dim=1)  # combine two images in channel dimension
        x = self.head(self.neck(self.backbone(x)))
        delta = x.view(-1, 4, 2)
        return delta


    def forward_train(self, x):
        patch_a = x['patch_a']
        patch_b = x['patch_b']
        output = torch.cat((patch_a, patch_b), dim=1)
        output = self.head(self.neck(self.backbone(output)))
        delta = output.view(-1, 4, 2)

        x['delta'] = delta
        loss = self.loss(x)

        return loss

