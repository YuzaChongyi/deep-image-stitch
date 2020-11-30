# _*_ coding: utf-8 _*_
# @Time : 2020/11/30 2:40 下午 
# @Author : Blue
# @File : collate.py

from torch.utils.data.dataloader import default_collate

def safe_collate(batch):
    """Return batch without any None values"""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)