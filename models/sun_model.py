import sys
import torch

import torch.nn as nn
import torch.nn.functional as F
from .conv_network import ResBlock
from .conv_network import ConvBlock
from .conv_network import BaseEncoderDecoder

from .semantic_embedding import SemanticEmbedding
from .mpi import ComputeHomography
from .mpi import AlphaComposition
from .mpi import ApplyHomography
from .mpi import Alpha2Disp
from .mpi import ApplyAssociation

class MulLayerConvNetwork(torch.nn.Module):
    
    def __init__(self, opts):
        super(MulLayerConvNetwork, self).__init__()
        self.opts = opts
        input_channels = opts.num_classes
        num_planes = opts.num_planes
        enc_features = opts.mpi_encoder_features
        self.input_channels = input_channels
        self.num_classes = opts.num_classes
        self.num_planes = num_planes
        self.out_seg_chans = self.opts.embedding_size
        self.discriptor_net = BaseEncoderDecoder(input_channels)
        self.base_res_layers = nn.Sequential(*[ResBlock(enc_features, 3) for i in range(2)])
        total_seg_channels = (self.opts.num_layers-1)*self.out_seg_chans # we will re-use the input semantics
        total_alpha_channels = num_planes
        self.total_seg_channels = total_seg_channels
        self.total_alpha_channels = total_alpha_channels
        self.total_beta_channels = num_planes*self.opts.num_layers
        total_output_channels = total_seg_channels + total_alpha_channels + self.total_beta_channels
        self.blending_alpha_seg_beta_pred = nn.Sequential(ResBlock(enc_features, 3),
                                ResBlock(enc_features, 3),
                                nn.SyncBatchNorm(enc_features),
                                ConvBlock(enc_features, total_output_channels//2, 3, down_sample=False),
                                nn.SyncBatchNorm(total_output_channels//2),
                                ConvBlock(total_output_channels//2,
                                            total_output_channels, 3,
                                            down_sample=False,
                                            use_no_relu=True))

    def forward(self, input_sem):
        # input_sem = data['input_seg']
        b, _, h, w = input_sem.shape
        feats_0 = self.discriptor_net(input_sem)
        feats_1 = self.base_res_layers(feats_0)
        alpha_and_seg_beta = self.blending_alpha_seg_beta_pred(feats_1)
        alphas = alpha_and_seg_beta[:, -self.total_alpha_channels:, :, :]
        seg = alpha_and_seg_beta[:, self.total_beta_channels:self.total_beta_channels+self.total_seg_channels, :, :]
        beta = alpha_and_seg_beta[:, :self.total_beta_channels, :, :]
        alpha = alphas.view(b, self.num_planes, 1, h, w)
        seg = seg.view(b, (self.opts.num_layers-1), self.out_seg_chans, h, w)
        beta = beta.view(b, self.num_planes, self.opts.num_layers, h, w)
        return alpha, seg, beta


class SUNModel(torch.nn.Module):
    '''
        A wrapper class for predicting MPI and doing rendering
    '''
    def __init__(self, opts):
        super(SUNModel, self).__init__()
        self.opts = opts
        self.conv_net = MulLayerConvNetwork(opts)
        self.compute_homography = ComputeHomography(opts)
        self.alpha_composition = AlphaComposition()
        self.apply_homography = ApplyHomography()
        self.alpha_to_disp = Alpha2Disp(opts)
        self.apply_association = ApplyAssociation(opts.num_layers)
        if not (opts.embedding_size==opts.num_classes):
            self.semantic_embedding = SemanticEmbedding(num_classes=opts.num_classes,
                                                        embedding_size=opts.embedding_size)


    def forward(self, input_data):
        # return self.conv_net(input_dict)
        input_seg = input_data['input_seg']
        # k_matrix = input_data['k_matrix']
        # t_vec, r_mat = input_data['t_vec'], input_data['rot_mat']
        encoding_needed = not (self.opts.num_classes==self.opts.embedding_size)
        input_seg_ = self.semantic_embedding.encode(input_seg) if encoding_needed else input_seg
        # Compute MPI alpha, multi layer semantics and layer to plane associations
        alpha, seg_mul_layer, associations = self.conv_net(input_seg_)
        # Append the input semantics to the multi layer semantics
        seg_mul_layer = F.softmax(seg_mul_layer, dim=2)
        seg_mul_layer = torch.cat([input_seg_.unsqueeze(1), seg_mul_layer], dim=1)
        # torch.cat([input_seg_.unsqueeze(1), F.softmax(seg_mul_plane, dim=2)], dim=1)
        alpha = torch.sigmoid(torch.clamp(alpha, min=-100, max=100))
        associations = F.softmax(associations, dim=2)
        return seg_mul_layer, alpha, associations
