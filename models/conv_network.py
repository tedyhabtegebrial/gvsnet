import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """ docstring for ResBlock."""

    def __init__(self, in_ch, k):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_ch, in_ch//2, kernel_size=k, stride=1, padding=k // 2)
        self.conv_2 = nn.Conv2d(in_ch // 2, in_ch, kernel_size=k, stride=1, padding=k // 2)
    def forward(self, input_):
        out_1 = F.relu(self.conv_1(input_))
        out_2 = F.relu(self.conv_2(out_1))
        out = out_2 + input_
        return out

class ConvBlock(nn.Module):
    def __init__(self, inp_chans, out_chans, k_size, down_sample=True, use_no_relu=False):
        super(ConvBlock, self).__init__()
        stride_0 = 2 if down_sample else 1
        self.conv_0 = nn.Conv2d(inp_chans, out_chans, kernel_size=k_size, stride=stride_0, padding=k_size // 2)
        self.conv_1 = nn.Conv2d(out_chans, out_chans, kernel_size=k_size, stride=1, padding=k_size // 2)
        nn.init.xavier_normal_(self.conv_0.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.conv_1.weight.data, gain=1.0)
        self.non_linearity_0 = F.relu
        self.non_linearity_1 = F.relu
        if use_no_relu:
            self.non_linearity_1 = nn.Sequential()
        else:
            self.non_linearity_1 = F.relu

    def forward(self, x):
        x1 = self.non_linearity_0(self.conv_0(x))
        x2 = self.non_linearity_1(self.conv_1(x1))
        return x2

class DeconvBlock(nn.Module):
    def __init__(self, inp_chans, out_chans):
        super(DeconvBlock, self).__init__()
        self.conv_0 = nn.Conv2d(inp_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.non_linearity = nn.ReLU(True)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1) if isinstance(inputs, list) else inputs
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv_0(x))
        return x

class BaseEncoderDecoder(torch.nn.Module):
    """ This class serves as the base encoder decoder network usef for estimating"""

    def __init__(self, in_chans, feat_size=32, output_feats=96, kernel_size=3):
        super(BaseEncoderDecoder, self).__init__()
        self.conv_block_1 = ConvBlock(in_chans, feat_size, 5)  # H/2
        self.conv_block_2 = ConvBlock(feat_size, int(2 * feat_size), 5)  # H/4
        self.conv_block_3 = ConvBlock(int(2 * feat_size), int(4 * feat_size), kernel_size)  # H/8
        self.conv_block_4 = ConvBlock(int(4 * feat_size), int(8 * feat_size), kernel_size)  # H/16
        self.conv_block_5 = ConvBlock(int(8 * feat_size), int(16 * feat_size), kernel_size)  # H/32
        self.conv_block_6 = ConvBlock(int(16 * feat_size), int(16 * feat_size), kernel_size)  # H/64
        self.conv_block_7 = ConvBlock(int(16 * feat_size), int(16 * feat_size), kernel_size)  # H/128
        self.up_conv_7 = DeconvBlock(int(16 * feat_size), int(16 * feat_size))          # H/64
        self.up_conv_6 = DeconvBlock(int(32 * feat_size), int(16 * feat_size))          # H/8
        self.up_conv_5 = DeconvBlock(int(32 * feat_size), int(16 * feat_size))          # H/8
        self.up_conv_4 = DeconvBlock(int(24 * feat_size), int(12 * feat_size))          # H/8
        self.up_conv_3 = DeconvBlock(int(16 * feat_size), int(8 * feat_size))           # H/8
        self.up_conv_2 = DeconvBlock(int(10 * feat_size), int(3 * feat_size))           # H/8
        self.up_conv_1 = DeconvBlock(int(4 * feat_size), int(3 * feat_size))            # H/4
        self.out_conv = ConvBlock(int(3 * feat_size), output_feats, 3,
                                        down_sample=False)            # H/2

    def forward(self, input):
        b_1 = self.conv_block_1(input)
        b_2 = self.conv_block_2(b_1)
        b_3 = self.conv_block_3(b_2)
        b_4 = self.conv_block_4(b_3)
        b_5 = self.conv_block_5(b_4)
        b_6 = self.conv_block_6(b_5)
        b_7 = self.conv_block_7(b_6)

        u_7 = self.up_conv_7(b_7)
        u_6 = self.up_conv_6([u_7, b_6])
        u_5 = self.up_conv_5([u_6, b_5])
        u_4 = self.up_conv_4([u_5, b_4])
        u_3 = self.up_conv_3([u_4, b_3])
        u_2 = self.up_conv_2([u_3, b_2])
        u_1 = self.up_conv_1([u_2, b_1])
        out = self.out_conv(u_1)
        return out

class ConvNetwork(torch.nn.Module):

    def __init__(self, opts):
        super(ConvNetwork, self).__init__()
        if not(opts.num_classes==opts.embedding_size):
            embedding_size = opts.embedding_size
            opts.__dict__['num_classes'] = embedding_size
            # opts = {**opts, 'num_classes':embedding_size}
        self.opts = opts
        input_channels = opts.embedding_size
        num_planes = opts.num_planes
        enc_features = opts.mpi_encoder_features
        self.input_channels = input_channels
        self.num_classes = opts.num_classes
        self.num_planes = num_planes
        self.out_seg_chans = self.num_classes
        self.discriptor_net = BaseEncoderDecoder(input_channels)
        self.base_res_layers = nn.Sequential(*[ResBlock(enc_features, 3) for i in range(2)])
        self.blending_alpha_seg_pred = nn.Sequential(ResBlock(enc_features, 3),
                                ResBlock(enc_features, 3),
                                nn.SyncBatchNorm(enc_features),
                                ConvBlock(enc_features, int(num_planes*(self.out_seg_chans+1))//2, 3, down_sample=False),
                                nn.SyncBatchNorm(int(num_planes*(self.out_seg_chans+1))//2),
                                ConvBlock(int(num_planes*(self.out_seg_chans+1))//2,
                                            int(num_planes*(self.out_seg_chans+1)), 3,
                                            down_sample=False,
                                            use_no_relu=True))

    def forward(self, input_img):
        b, nc, h, w = input_img.shape
        feats_0 = self.discriptor_net(input_img)
        feats_1 = self.base_res_layers(feats_0)
        alpha_and_seg = self.blending_alpha_seg_pred(feats_1)
        alpha_and_seg = alpha_and_seg.view(b, self.num_planes, self.out_seg_chans+1, h, w)
        return alpha_and_seg
