import math
import torch
import time
from contextlib import contextmanager


@contextmanager
def dummy_progress_bar():
    yield None

def lr_func(num_epochs):
    def func(s):
        if s < (num_epochs//2):
            return 1
        else:
            return max(0, 1-(2*s-num_epochs)/num_epochs)
    return func

def worker_init_fun(worker_id):
    torch.random.manual_seed(12345)
    np.random.seed(12345)
    random.seed(12345)

def get_current_time():
    time_struct = time.gmtime()
    # construct a folder name from the current time
    folder_name = 'y_'
    folder_name += str(time_struct.tm_year) + '_d_'
    folder_name += str(time_struct.tm_yday).zfill(3) + '_h_'
    folder_name += str(time_struct.tm_hour).zfill(2) + \
        '_m_' + str(time_struct.tm_min).zfill(2)
    return folder_name

def convert_model(module):

    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.SynchBatchNorm, to torch.nn.BatchNorm2d
    Borrowed from https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/blob/master/sync_batchnorm/batchnorm.py
    and modified
    """
    mod = module
    if isinstance(module, (torch.nn.SyncBatchNorm, )):
        mod = torch.nn.BatchNorm2d(module.num_features, module.eps,
                             module.momentum, module.affine)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))
    return mod


def get_cam_poses(movement_type='circle', b_size=1, num_cameras=10):
    if movement_type == 'circle':
        r = 0.25
        angles = torch.linspace(0.0, 2*math.pi, num_cameras)
        x_locs = -1*r*torch.cos(angles)
        y_locs = -1*r*torch.sin(angles)
        z_locs = -1*0.2*torch.ones(num_cameras)
    elif movement_type == 'lateral':
        x_locs = list(torch.linspace(-0.25, 0.25, 10))
        x_locs = x_locs + list(reversed(x_locs))[1:-1]
        x_locs = torch.FloatTensor(x_locs)
        y_locs = torch.zeros_like(x_locs)
        z_locs = torch.ones_like(x_locs)
    else:
        raise NotImplementedError(f'unknown camera movement {movement_type}')
    num_cameras = len(x_locs)
    t_vecs = torch.stack([x_locs, y_locs, z_locs], dim=1).view(num_cameras, 3)
    t_vecs = t_vecs.view(1, num_cameras, 3, 1).expand(
        b_size, num_cameras, 3, 1)
    t_vecs = torch.split(t_vecs, split_size_or_sections=1, dim=1)
    t_vecs = [t.view(b_size, 3, 1) for t in t_vecs]
    # rotation matrices will be indetity
    r_mats = [torch.eye(3).view(1, 3, 3) for t in t_vecs]
    r_mats = [torch.eye(3).expand(b_size, 3, 3) for t in t_vecs]
    return t_vecs, r_mats
