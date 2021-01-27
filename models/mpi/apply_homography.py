import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F


class ApplyHomography(nn.Module):

    def __init__(self):
        super(ApplyHomography, self).__init__()

    def get_xy_locations(self, num_imgs, h, w, homogenous=True):
        x_range, y_range = map(lambda x: torch.linspace(0, x - 1, x), [w, h])
        x_range, y_range = x_range.view(1, 1, w, 1), y_range.view(1, h, 1, 1)
        # device_ = torch.device(f'cuda:{self.opts.local_rank}')
        # x_range, y_range = x_range.to(device_), y_range.to(device_)
        x_range, y_range = x_range.expand(num_imgs, h, w, 1), \
                           y_range.expand(num_imgs, h, w, 1)
        xy_locs = torch.cat([x_range, y_range], 3)
        if homogenous:
            ones = torch.ones_like(x_range)
            xy_locs = torch.cat([xy_locs, ones], 3)
        return xy_locs.unsqueeze(-1)

    def get_grid(self, h_matrix, src_img, multiplane=True, mode='bilinear'):
        device_ = h_matrix.device
        num_d = h_matrix.shape[1]
        if not multiplane:
            b_size, f_size, h, w = src_img.shape
            src_img = src_img.unsqueeze(1).expand(b_size, num_d, f_size, h, w)
            src_img = src_img.contiguous()
        b_size, _num_d, f_size, h, w = src_img.shape
        src_img = src_img.view(-1, f_size, h, w)
        h_matrix = h_matrix.view(-1, 3, 3)
        xy_locations = self.get_xy_locations(b_size * num_d, h, w, homogenous=True).to(device_)
        xy_locations = xy_locations.view(b_size * num_d, h, w, 3, 1).view(-1, 3, 1)
        h_matrix = h_matrix.view(-1, 3, 3).view(b_size * num_d, 1, 1, 3, 3)
        h_matrix = h_matrix.expand(b_size * num_d, h, w, 3, 3)
        h_matrix = h_matrix.contiguous().view(-1, 3, 3)
        warped_locs = torch.split(torch.bmm(h_matrix, xy_locations),
                                  split_size_or_sections=1,
                                  dim=1)
        x_locs = warped_locs[0] / warped_locs[2]
        x_locs = (x_locs - ((w-1.0) / 2.0)) / ((w-1.0) / 2.0)
        y_locs = warped_locs[1] / warped_locs[2]
        y_locs = (y_locs - ((h-1.0) / 2.0)) / ((h-1.0) / 2.0)
        grid = torch.cat([x_locs, y_locs], dim=2)
        grid = grid.view(b_size, num_d, h, w, 2)
        return grid
    def warp_images(self, h_matrix, src_img, multiplane=True, mode='bilinear', grid=None):
        # print(h_matrix.shape, src_img.shape)
        # exit()
        b_size, num_d, f_size, h, w = src_img.shape
        if grid is None:
            grid = self.get_grid(h_matrix, src_img, multiplane=True)
        grid_ = grid.view(-1, h, w, 2)
        # print(src_img.shape, grid.shape)
        src_img = src_img.contiguous().view(-1, f_size, h, w)
        # exit()
        warped_views = F.grid_sample(input=src_img, grid=grid_, mode=mode)
        warped_views = warped_views.view(b_size, num_d, f_size, h, w)
        return warped_views, grid
    def apply_grid(self, src_img, grid, mode='bilinear'):
        b_size, num_d, f_size, h, w = src_img.shape
        grid_ = grid.view(-1, h, w, 2)
        src_img = src_img.contiguous().view(-1, f_size, h, w)
        warped_views = F.grid_sample(input=src_img, grid=grid_, mode=mode, align_corners=False)
        warped_views = warped_views.view(b_size, num_d, f_size, h, w)
        return warped_views

    def forward(self, h_matrix, src_img, mode='bilinear', grid=None):
        assert h_matrix.ndimension() == 4, 'H Matrix should be a 4D Tensor'
        assert src_img.ndimension() >= 4, "src should be atleast 4D Tensor "
        if src_img.ndimension() == 4:
            return self.warp_images(h_matrix, src_img, multiplane=False, mode=mode, grid=grid)
        elif src_img.ndimension() == 5:
            return self.warp_images(h_matrix, src_img, multiplane=True, mode=mode, grid=grid)
        else:
            assert False, "Wrong shape of src image"