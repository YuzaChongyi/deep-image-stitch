# _*_ coding: utf-8 _*_
# @Time : 2020/11/30 1:48 下午 
# @Author : Blue
# @File : linear_head.py

import torch.nn as nn
from ..utils.weight_init import normal_init

class LinearHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout = None):
        super(LinearHead, self).__init__()

        assert isinstance(num_classes, (int, list))
        if isinstance(num_classes, int):
            num_classes = [num_classes]
        for _num_classes in num_classes:
            assert _num_classes > 0, 'num_classes should be larger than 0'

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes[0])
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.init_weights()

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        return self.fc(x)

