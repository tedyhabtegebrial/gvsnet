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

    def log_depth(self, input_dict):
        input_dict = {k: self.reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = v / v.amax(dim=(1,2,3), keepdim=True)
            self.writer.add_image(f'depth/{k}', make_grid(v), self.iteration)

    def log_images(self, input_dict):
        input_dict = {k: self.reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            v = (v + 1)/2.0
            self.writer.add_image(f'color/{k}', make_grid(v), self.iteration)

    def log_scalar(self, input_dict):
        for k,v in input_dict.items():
            v = v if isinstance(v, (float, int)) else v.item()
            self.writer.add_scalar(f'scalar/{k}', v, self.iteration)

    def log_semantics(self, input_dict):
        input_dict = {k: self.reshape(v) for k, v in input_dict.items()}
        for k, v in input_dict.items():
            if v.shape[1]>1:
                v = v.argmax(dim=1, keepdim=True)
            imgs = [torch.from_numpy(self.save_semantics.to_color(im)) for im in v]
            imgs = make_grid(torch.stack(imgs))
            self.writer.add_image(f'semantics/{k}', imgs, self.iteration)
    
    def step(self): 
        self.iteration += 1

    def reshape(self, x):
        return x.view(-1, -1, x.shape[-2], x.shape[-1])
