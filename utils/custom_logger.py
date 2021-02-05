import os
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
        self.save_semantics = SaveSemantics(dataset)
        self.writer = SummaryWriter(logging_path)
        self.iteration = 0
    def amax(self, x, dim=(-1,)):
        for d in dim:
            x_max = x.max(d, keepdim=keepdim)



    def log_depth(self, input_dict):
        def reshape(x): return x.view(-1, 1, x.shape[-2], x.shape[-1])
        input_dict = {k: reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = v / torch.max(v)
            self.writer.add_image(f'depth/{k}', make_grid(v), self.iteration)

    def log_images(self, input_dict):
        def reshape(x): return x.view(-1, 3, x.shape[-2], x.shape[-1])
        input_dict = {k: reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = (v + 1)/2.0
            self.writer.add_image(f'color/{k}', make_grid(v), self.iteration)

    def log_scalar(self, input_dict):
        for k,v in input_dict.items():
            v = v if isinstance(v, (float, int)) else v.item()
            self.writer.add_scalar(f'scalar/{k}', v, self.iteration)

    def log_semantics(self, input_dict):
        reshape = lambda x: x.view(-1, self.num_classes, x.shape[-2], x.shape[-1])
        input_dict = {k: reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            if v.shape[1]>1:
                v = v.argmax(dim=1, keepdim=True)
            imgs = [torch.from_numpy(copy(self.save_semantics.to_color(im))) for im in v]
            imgs = make_grid(torch.stack(imgs))
            self.writer.add_image(f'semantics/{k}', imgs, self.iteration)
    
    def step(self): 
        self.iteration += 1


