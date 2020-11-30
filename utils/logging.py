# _*_ coding: utf-8 _*_
# @Time : 2020/11/30 2:42 下午 
# @Author : Blue
# @File : logging.py

import logging

def initial_logger():
    FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger
