# _*_ coding: utf-8 _*_
# @Time : 2020/11/30 2:46 下午 
# @Author : Blue
# @File : synthetic.py

from pathlib import Path
import kornia
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SyntheticDataset(Dataset):
    def __init__(self, folder, img_size=512, patch_size=256, rho=45, filetypes=[".jpg", ".jpeg", ".png"]):
        super(SyntheticDataset, self).__init__()
        self.fnames = []
        for filetype in filetypes:
            self.fnames.extend(list(Path(folder).glob(f"*{filetype}")))
        self.transforms = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.rho = rho

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_a = Image.open(self.fnames[index]).convert('RGB')
        img_a = self.transforms(img_a)

        # grayscale
        img_a = torch.mean(img_a, dim=0, keepdim=True)

        # pick top left corner
        x = random.randint(self.rho, self.img_size - self.rho - self.patch_size)
        y = random.randint(self.rho, self.img_size - self.rho - self.patch_size)

        corners = torch.tensor(
            [
                [x, y],
                [x + self.patch_size, y],
                [x + self.patch_size, y + self.patch_size],
                [x, y + self.patch_size],
            ]
        )
        delta = torch.randint_like(corners, -self.rho, self.rho)
        perturbed_corners = corners + delta

        try:
            # compute homography from points
            h = kornia.get_perspective_transform(
                corners.unsqueeze(0).float(), perturbed_corners.unsqueeze(0).float()
            )

            h_inv = torch.inverse(h)

            # apply homography to single img
            img_b = kornia.warp_perspective(img_a.unsqueeze(0), h_inv, (self.img_size, self.img_size))[0]

        except:
            # either matrix could not be solved or inverted
            # this will show up as None, so use safe_collate in train.py
            return None

        patch_a = img_a[:, y : y + self.patch_size, x : x + self.patch_size]
        patch_b = img_b[:, y : y + self.patch_size, x : x + self.patch_size]

        return img_a, patch_a, patch_b, corners.float(), delta.float()
