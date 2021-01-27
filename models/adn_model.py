import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class Conv2DBlock(nn.Module):
    def __init__(self, in_chans, out_chans, use_tanh=False, use_spectral_norm=True):
        super(Conv2DBlock, self).__init__()
        self.use_tanh = use_tanh
        conv_1 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, padding=1)
        if use_spectral_norm:
            self.conv_1 = nn.Sequential(spectral_norm(conv_1))
        else:
            self.conv_1 = nn.Sequential(conv_1)
    def actvn(self, x):
        if self.use_tanh:
            return F.tanh(x)
        else:
            return F.leaky_relu(x, negative_slope=2e-1)

    def forward(self, x):
        return self.actvn(self.conv_1(x))

class AppearaceDecoderNetwork(nn.Module):
    def __init__(self, num_inp_chans):
        super(AppearaceDecoderNetwork, self).__init__()
        feat_sizes = [num_inp_chans, 16, 32, 32, 64, 64]
        self.layer_0 = Conv2DBlock(in_chans=feat_sizes[0], out_chans=feat_sizes[1], use_spectral_norm=True)
        self.layer_1 = Conv2DBlock(in_chans=feat_sizes[1], out_chans=feat_sizes[2], use_spectral_norm=True)
        self.layer_2 = Conv2DBlock(in_chans=feat_sizes[2], out_chans=feat_sizes[3], use_spectral_norm=True)
        self.layer_3 = Conv2DBlock(in_chans=feat_sizes[3], out_chans=feat_sizes[4], use_spectral_norm=True)
        self.layer_4 = Conv2DBlock(in_chans=feat_sizes[4], out_chans=feat_sizes[5], use_spectral_norm=True)

        self.decoder_4 = Conv2DBlock(in_chans=feat_sizes[5], out_chans=feat_sizes[4], use_spectral_norm=True)
        self.decoder_3 = Conv2DBlock(in_chans=feat_sizes[4] + feat_sizes[4], out_chans=feat_sizes[3], use_spectral_norm=True)
        self.decoder_2 = Conv2DBlock(in_chans=feat_sizes[3] + feat_sizes[3], out_chans=feat_sizes[2], use_spectral_norm=True)
        self.decoder_1 = Conv2DBlock(in_chans=feat_sizes[2] + feat_sizes[2], out_chans=feat_sizes[1], use_spectral_norm=True)
        self.output_conv = Conv2DBlock(in_chans=feat_sizes[1] + feat_sizes[1], out_chans=3, use_tanh=True, use_spectral_norm=False)

    def dn(self, x):
        return F.interpolate(x, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True)

    def up(self, x):
        return F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)

    def forward(self, input_imgs):
        s0 = self.layer_0(input_imgs)
        s1 = self.dn(self.layer_1(s0))
        s2 = self.dn(self.layer_2(s1))
        s3 = self.dn(self.layer_3(s2))
        s4 = self.dn(self.layer_4(s3))
        u4 = self.decoder_4(self.up(s4))
        u3 = self.decoder_3(self.up(torch.cat([u4, s3], dim=1)))
        u2 = self.decoder_2(self.up(torch.cat([u3, s2], dim=1)))
        u1 = self.decoder_1(self.up(torch.cat([u2, s1], dim=1)))
        u0 = self.output_conv(torch.cat([u1, s0], dim=1))
        return u0
