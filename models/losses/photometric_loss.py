# _*_ coding: utf-8 _*_
# @Time : 2020/11/30 2:13 下午 
# @Author : Blue
# @File : photometric_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class PhotometricLoss(nn.Module):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        super(PhotometricLoss, self).__init__()

    def forward(self, data):
        corners = data['corners']
        delta = data['delta']
        img_a = data['img_a']
        patch_b = data['patch_b']

        corners_hat = corners + delta

        # in order to apply transform and center crop,
        # subtract points by top-left corner (corners[N, 0])
        corners = corners - corners[:, 0].view(-1, 1, 2)

        h = kornia.get_perspective_transform(corners, corners_hat)

        h_inv = torch.inverse(h)
        patch_b_hat = kornia.warp_perspective(img_a, h_inv, (self.patch_size, self.patch_size))

        return F.l1_loss(patch_b_hat, patch_b)

