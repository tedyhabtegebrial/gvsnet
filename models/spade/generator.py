"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_network import BaseNetwork
from .normalization import get_nonspade_norm_layer
from .architecture import ResnetBlock as ResnetBlock
from .architecture import SPADEResnetBlock as SPADEResnetBlock

class SPADEGenerator(BaseNetwork):
    def __init__(self, opt, no_tanh=True):
        '''
            Since we extract layered features 
        '''
        super().__init__()
        # opt = self.modify_commandline_options(opt, True)
        self.opt = opt
        self.no_tanh = no_tanh
        num_context_chans = 0
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            if opt.use_instance_mask:
                self.fc = nn.Conv2d(self.opt.embedding_size+1, 16 * nf, 3, padding=1)
            else:
                self.fc = nn.Conv2d(self.opt.embedding_size, 16 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock((16 * nf), 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock((16 * nf), 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock((16 * nf), 16 * nf, opt)

        self.up_0 = SPADEResnetBlock((16 * nf), 8 * nf, opt)
        self.up_1 = SPADEResnetBlock((8 * nf), 4 * nf, opt)
        self.up_2 = SPADEResnetBlock((4 * nf), 2 * nf, opt)
        self.up_3 = SPADEResnetBlock((2 * nf), 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock((1 * nf), nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, opt.num_out_channels, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        # self.init_weights('xavier', gain=0.02)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.width // (2**num_up_layers)
        sh = opt.height // (2**num_up_layers)

        return sw, sh

    def forward(self, seg_map, z=None):
        seg = seg_map
        if self.opt.use_vae:
            x = self.fc(z)
        else:
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
        x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)
        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if self.no_tanh:
            return x
        else:
            x = torch.tanh(x)
            return x
