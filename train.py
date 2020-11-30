# _*_ coding: utf-8 _*_
# @Time : 2020/11/30 2:21 下午 
# @Author : Blue
# @File : train.py

import os
import time
import yaml
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from models.applications.UDH import UDH
from datasets.utils.collate import safe_collate
from datasets.synthetic import SyntheticDataset
from utils.meter import AverageMeter

def train(train_loader,
          model,
          optimizer,
          epoch,
          device='cpu',
          print_freq=100,
          parallel=False,
          save_prefix='checkpoint',
          save_path=None
          ):
    loss_total = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    model.train()

    for i, (img_a, patch_a, patch_b, corners, gt_delta) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img_a, patch_a, patch_b, corners, delta = \
            img_a.to(device), patch_a.to(device), patch_b.to(device), corners.to(device), gt_delta.to(device)

        model_inp = {'img_a': img_a, 'patch_a': patch_a, 'patch_b':patch_b, 'corners': corners}
        loss = model.forward_train(model_inp)
        loss_total.update(loss.item(), img_a.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{}][{}/{}] time: {:.3f} loss_total: {:.4f}'.format(epoch, i, len(train_loader), batch_time.avg, loss_total.avg))

    print('Epoch: [{}] Time cost: {:.3f}'.format(epoch, batch_time.sum))

    if save_path:
        state_dict = model.module.state_dict() if parallel else model.state_dict()
        torch.save(state_dict, os.path.join(save_path, '%s_epoch_%d.pth' % (save_prefix, epoch)))


if __name__ == '__main__':
    with open('config/udh.yml', 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)

    train_imgs_folder = data['TrainDataset']['img_dir']
    val_imgs_folder =  data['ValDataset']['img_dir']
    resume = data['Train']['resume']


    parallel = False
    model = UDH('efficientnet-b0', patch_size=128)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if data['Train']['resume']:
        model.load_state_dict(torch.load(data['Train']['resume_path'], map_location=device))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_dataset = SyntheticDataset(train_imgs_folder, img_size=256, patch_size=128)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=safe_collate)

    for epoch in range(5):
        train(train_loader, model, optimizer,
              epoch=epoch, device=device, print_freq=1, parallel=False, save_path=data['Train']['checkpoint_path'])
        scheduler.step()



