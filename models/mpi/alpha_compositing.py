import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaComposition(nn.Module):

    def __init__(self):
        super(AlphaComposition, self).__init__()
    def forward(self, src_imgs, alpha_imgs):
        b_size, num_d, _c, h, w = src_imgs.shape
        src_imgs = torch.split(src_imgs, split_size_or_sections=1, dim=1)
        alpha_imgs = torch.split(alpha_imgs, split_size_or_sections=1, dim=1)
        comp_rgb = src_imgs[-1] # * alpha_imgs[-1]
        for d in reversed(range(num_d - 1)):
            comp_rgb = src_imgs[d] * alpha_imgs[d] + (1.0 - alpha_imgs[d]) * comp_rgb
        return comp_rgb.squeeze(1)

class Alpha2Disp(nn.Module):
    def __init__(self, opts):
        super(Alpha2Disp, self).__init__()
        self.depth_proposals = 1 / torch.linspace(1 / opts.near_plane, 1 / opts.far_plane, opts.num_planes)
        self.depth_proposals = self.depth_proposals.view(opts.num_planes)
        self.opts = opts

    def forward(self, alpha, k_mat, baseline, t_vec=None, novel_view=False):
        '''converts input-view or novel-view alpha to disparity
            we assume there is no rotation between the input and novel views
        '''
        device_ = alpha.device
        batch_size = alpha.shape[0]
        depth_proposals = self.depth_proposals.view(1, -1).clone()
        depth_proposals = depth_proposals.expand(
            batch_size, self.opts.num_planes).to(device_)
        b_size, num_d, _c, _h, _w = alpha.shape
        alpha = torch.split(alpha, split_size_or_sections=1, dim=1)
        if novel_view:
            z_shift = t_vec[:, 2, :].view(b_size, 1)
        else:
            z_shift = torch.zeros(b_size, 1).to(device_)
        depth_proposals = depth_proposals + z_shift  
        depth_proposals = torch.split(depth_proposals, split_size_or_sections=1, dim=1)
        disp_proposals = [baseline*(k_mat[:, 0, 0]).view(-1, 1, 1, 1, 1)/d.view(batch_size, 1, 1, 1, 1) for d in depth_proposals]
        disp_map = disp_proposals[-1]# * alpha[-1]
        for d in reversed(range(num_d - 1)):
            disp_map = disp_proposals[d] * alpha[d] + (1.0 - alpha[d]) * disp_map
        return disp_map.squeeze(1)