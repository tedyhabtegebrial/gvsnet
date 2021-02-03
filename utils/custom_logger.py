import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .save_results import SaveSemantics

class Logger(object):
    def __init__(self, logging_path='', dataset='carla'):
        super().__init__()
        assert os.path.exists(logging_path), 'pass logging path'
        self.logging_path = logging_path
        self.num_classes = 13
        self.writer = SummaryWriter(logging_path)
        self.iteration = 0

    def log_images(self, input_dict, prefix='color'):
        input_dict = {k: self.reshape(v) for k, v in input_dict.items()}
        input_dict = {k: make_grid(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            self.writer(f'{prefix}/{k}', v, self.iteration)
    
    def log_semantics(self, input_dict):
        input_dict = {k: self.reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            if v.shape[1]>1:
                v = v.argmax(dim=1, keepdim=True)
            imgs = [torch.from_numpy(self.save_semantics.to_color(im)) for im in v]
            imgs = make_grid(torch.stack(imgs))
            self.writer(f'semantics/{k}', imgs, self.iteration)
    
    def step(self): 
        self.iteration += 1

    def reshape(self, x):
        return x.view(-1, -1, x.shape[-2], x.shape[-1])
