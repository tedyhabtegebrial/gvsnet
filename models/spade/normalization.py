"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        # print(subnorm_type)
        # print(subnorm_type)
        # exit()
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            # synch batch norm is dropped in favor of pytorch's synch_batch_norm utility
            norm_layer = nn.SyncBatchNorm(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, k_size, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)

        nhidden = min(norm_nc, 128)

        pw = k_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=k_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=k_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=k_size, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

def add_normalisation_layer(input_layer, normalisation, spectral):
    assert isinstance(input_layer, nn.Conv2d), f'layer should be {nn.Conv2d}'
    layers = []
    if spectral:
        layers.append(spectral_norm(input_layer))
    else:
        layers.append(input_layer)
    if normalisation == 'instance':
        norm_layer = nn.InstanceNorm2d(input_layer.out_channels, affine=False)
    elif normalisation == 'batch':
        norm_layer = nn.BatchNorm2d(input_layer.out_channels, affine=True)
    else:
        raise ValueError(f'Unspecified nomalisation layer {normalisation}')
    layers.append(norm_layer)
    return layers

if __name__=='__main__':
    k_size, norm_nc, label_nc = 3, 32, 13
    spade = SPADE(k_size, norm_nc, label_nc)
    x = torch.rand(1,32,32,32)
    labels = torch.rand(1,13,256,256)
    out = spade(x, labels)
    print(x.shape)
