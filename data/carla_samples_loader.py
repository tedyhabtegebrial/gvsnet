import os
import math
import pathlib
import functools
import warnings

import cv2 as cv
from PIL import Image
import numpy as np
from skimage import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class LoadSamples(Dataset):
    def __init__(self, opts):
        super(LoadSamples, self).__init__()
        self.opts = opts
        assert os.path.exists(self.opts.data_path)
        files = sorted([f.__str__() for f in pathlib.Path(self.opts.data_path).rglob('*.png')])
        sem_files = [f for f in files if f.endswith('semantics.png')]
        assert len(
            sem_files) > 0, f'It seems like the folder {self.opts.data_path} contains no semantic maps'
        self.sem_files = sem_files
        col_files = [f for f in files if 'color.png' in f]
        self.col_files = col_files if len(col_files)>0 else None
        
        if opts.dataset == 'carla':
            self.reader = CarlaMiniLoader(opts)

    def __getitem__(self, index):
        data_dict = {}
        data_dict['input_seg'] = self.read_semantics(self.sem_files[index])
        if os.path.exists(self.opts.style_path):
            # we will use the external image as style
            data_dict['style_img']  = self.read_color(self.opts.style_path)
        else:
            data_dict['style_img'] = self.read_color(self.col_files[index])
        data_dict['k_matrix'] = self.get_k_matrix(height=self.opts.height, width=self.opts.width)
        data_dict = {k:v.float() for k,v in data_dict.items()}
        return data_dict

    def __len__(self):
        return len(self.sem_files)

    def read_semantics(self, filename):
        sem_img = np.array(Image.open(filename).convert(
            mode="P"), dtype=np.uint8)
        sem_img = torch.from_numpy(sem_img).squeeze()
        assert sem_img.max() < self.opts.num_classes, 'semantic ids should be <= num_classes-1 '
        sem_img = F.interpolate(input=sem_img.float().unsqueeze(0).unsqueeze(
            0), size=(self.opts.height, self.opts.width), mode='nearest').squeeze(0)
        # Convert label to one hot
        labels = torch.zeros(self.opts.num_classes,
                             self.opts.height, self.opts.width)
        labels = labels.scatter_(dim=0, index=sem_img.long(), value=1.0)
        return labels

    def read_color(self, filename):
        img = io.imread(filename)[..., :3]
        img_np = cv.resize(img, (self.opts.width, self.opts.height)) / 255.0
        img_np = (2*img_np)-1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor

    def get_k_matrix(self, fov=90.0, height=600, width=800):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / \
            (2.0 * math.tan(fov * math.pi / 360.0))
        return torch.from_numpy(k)
