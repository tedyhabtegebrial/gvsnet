import torch
import torch.nn as nn
import torch.nn.functional as F

class ComputeHomography(nn.Module):

    def __init__(self, opts):
        super(ComputeHomography, self).__init__()
        self.opts = opts
        self.h, self.w = opts.height, opts.width
        self.num_depths = opts.num_planes
        self.depth_proposals = 1 / torch.linspace(1 / opts.near_plane, 1 / opts.far_plane, opts.num_planes)
        self.depth_proposals = self.depth_proposals.view(opts.num_planes)
        self.src_corner_pts = [torch.Tensor([(self.w - 1) * i, (self.h - 1) * j, 1]) for i in range(2) for j in range(2)]

    def get_homography_matrices(self, kmats, r_mats, t_vec):
        device_ = kmats.device
        batch_size = r_mats.shape[0]
        num_dep = self.num_depths
        r_mats = r_mats.view(batch_size, 1, 3, 3).expand(batch_size, num_dep, 3, 3)
        r_mats = r_mats.contiguous().view(-1, 3, 3)
        t_vec = t_vec.view(batch_size, 1, 3, 1).contiguous().expand(batch_size, num_dep, 3, 1)
        t_vec = t_vec.contiguous().view(-1, 3, 1)
        kinv = torch.stack([torch.inverse(k) for k in kmats])
        kmats = kmats.view(-1, 1, 3, 3).expand(batch_size, num_dep, 3, 3).contiguous()
        kinv = kinv.view(-1, 1, 3, 3).expand(batch_size, num_dep, 3, 3).contiguous()
        kinv, kmats = kinv.view(-1, 3, 3), kmats.view(-1, 3, 3)
        n = torch.Tensor([0, 0, 1]).view(1, 1, 3).expand(r_mats.shape[0], 1, 3)
        n = n.to(device_).float()
        depth_proposals = self.depth_proposals.view(1, num_dep, 1).to(device_)
        depth_proposals = depth_proposals.expand(batch_size, num_dep, 1).contiguous()
        depth_proposals = depth_proposals.view(-1, 1, 1)
        num_1 = torch.bmm(torch.bmm(torch.bmm(r_mats.permute(0, 2, 1), t_vec), n), r_mats.permute(0, 2, 1))
        den_1 = -depth_proposals - torch.bmm(torch.bmm(n, r_mats.permute(0, 2, 1)), t_vec)
        h_mats = torch.bmm(torch.bmm(kmats, (r_mats.permute(0, 2, 1) + (num_1 / den_1))), kinv)
        h_mats = h_mats.view(batch_size, num_dep, 3, 3)
        return h_mats

    def forward(self, kmats, r_mats, t_vecs):
        hmats_1 = self.get_homography_matrices(kmats, r_mats, t_vecs)
        return hmats_1