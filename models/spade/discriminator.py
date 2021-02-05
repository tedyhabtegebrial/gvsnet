"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .base_network import BaseNetwork
from .normalization import get_nonspade_norm_layer
from .util import find_class_in_module


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        for i in range(opts.num_D):
            subnetD = NLayerDiscriminator(opts)
            self.add_module('discriminator_%d' % i, subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opts.num_D x opts.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opts.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opts.ndf
        input_nc = self.compute_D_input_nc(opts)

        norm_layer = get_nonspade_norm_layer(opts, opts.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opts.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opts.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opts):
        input_nc = opts.embedding_size + opts.output_nc
        if opts.contain_dontcare_label:
            input_nc += 1
        if not opts.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opts.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
