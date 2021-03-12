"""
license: please refere to the SPADE repositry license
Copied from https://github.com/NVlabs/SPADE/blob/master/models/networks
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .base_network import BaseNetwork
from .normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opts.ngf
        norm_layer = get_nonspade_norm_layer(opts, opts.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(
            nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opts.crop_size >= 256:
            self.layer6 = norm_layer(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opts = opts
        # self.init_weights('xavier', gain=0.02)
        # self.init_weights('orthogonal')

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opts.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if self.opts.use_vae:
            logvar = self.fc_var(x)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
