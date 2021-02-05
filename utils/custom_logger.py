import os
import cv2 as cv
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from copy import copy
from .save_results import SaveSemantics

class Logger(object):
    def __init__(self, logging_path='', dataset='carla'):
        super().__init__()
        assert os.path.exists(logging_path), 'pass logging path'
        self.logging_path = logging_path
        self.num_classes = 13
        self.semantics_tool = SaveSemantics(dataset)
        self.writer = SummaryWriter(logging_path)
        self.iteration = 0
    def amax(self, x, dim=(-1,)):
        for d in dim:
            x_max = x.max(d, keepdim=keepdim)

    def log_depth(self, input_dict):
        input_dict = {k: v.squeeze() for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = v / torch.max(v)
            self.writer.add_image(
                f'depth/{k}', v, self.iteration, dataformats='HW')

    def save_depth(self, input_dict):
        input_dict = {k: v.squeeze() for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = np.asarray(v, dtype=np.float32)
            v = v / v.max()
            file_suffix = f'{str(self.iteration).zfill(8)}_{k}.png'
            fname = os.path.join(self.logging_path, 'images', file_suffix)
            v = np.asarray(v*255, np.uint8)
            cv.imwrite(fname, v)

    def log_images(self, input_dict):
        input_dict = {k: v.squeeze() for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = (v + 1)/2.0
            self.writer.add_image(f'color/{k}', v, self.iteration, dataformats='CHW')

    def save_images(self, input_dict):
        input_dict = {k: v.squeeze() for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = (v + 1)/2.0
            v = (v.permute(1,2,0) * 255).numpy()
            file_suffix = f'{str(self.iteration).zfill(8)}_{k}.png'
            fname = os.path.join(self.logging_path, 'images', file_suffix)
            cv.imwrite(fname, v[..., [2,1,0]])

    def log_scalar(self, input_dict):
        for k,v in input_dict.items():
            v = v if isinstance(v, (float, int)) else v.item()
            self.writer.add_scalar(f'scalar/{k}', v, self.iteration)

    def log_semantics(self, input_dict):
        reshape = lambda x: x.view(self.num_classes, x.shape[-2], x.shape[-1])
        input_dict = {k: reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = v.argmax(dim=0, keepdim=True)
            v_col = self.semantics_tool.to_color(v)
            self.writer.add_image(f'semantics/{k}',torch.from_numpy(v_col.copy())/255.0,
                                    self.iteration, dataformats='HWC')

    def save_semantics(self, input_dict):
        def reshape(x): return x.view(
            self.num_classes, x.shape[-2], x.shape[-1])
        input_dict = {k: reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = v.argmax(dim=0, keepdim=True)
            v_col = self.semantics_tool.to_color(v)
            file_suffix = f'{str(self.iteration).zfill(8)}_{k}.png'
            fname = os.path.join(self.logging_path, 'images', file_suffix)
            cv.imwrite(fname, v_col[..., [2, 1, 0]])

    def step(self): 
        self.iteration += 1


