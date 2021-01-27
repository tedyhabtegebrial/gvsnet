import torch
import math


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