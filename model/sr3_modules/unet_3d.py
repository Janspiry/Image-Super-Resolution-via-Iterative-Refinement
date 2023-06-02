"""
Taken from https://github.com/roserustowicz/crop-type-mapping/
Implementation by the authors of the paper :
"Semantic Segmentation of crop type in Africa: A novel Dataset and analysis of deep learning methods"
R.M. Rustowicz et al.

Slightly modified to support image sequences of varying length in the same batch.
"""

import torch
import torch.nn as nn


def conv_block(in_dim, middle_dim, out_dim):
    print(in_dim, middle_dim, out_dim)
    model = nn.Sequential(
        nn.Conv3d(in_dim, middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


def center_in(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True))
    return model


def center_out(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(in_dim),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
    return model


def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return model


class UNet3D(nn.Module):
    def __init__(self, in_channel, out_channel, channel_mults, image_size, feats=16, pad_value=None, zero_pad=True):
        super(UNet3D, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        channel_mults = [4,8,16,32] #channel_mults
        self.image_size = image_size

        self.pad_value = pad_value
        self.zero_pad = zero_pad

        # Downs
        self.en3 = conv_block(in_channel, feats * channel_mults[0], feats * channel_mults[0])
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en4 = conv_block(feats * channel_mults[0], feats * channel_mults[1], feats * channel_mults[1])
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.en5 = conv_block(feats * channel_mults[1], feats * channel_mults[2], feats * channel_mults[2])
        self.pool_5 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Mids
        self.center_in = center_in(feats * channel_mults[2], feats * channel_mults[3])
        self.center_out = center_out(feats * channel_mults[3], feats * channel_mults[2])

        # Ups
        self.dc5 = conv_block(feats * channel_mults[3], feats * channel_mults[2], feats * channel_mults[2])
        self.trans4 = up_conv_block(feats * channel_mults[2], feats * channel_mults[1])
        self.dc4 = conv_block(feats * channel_mults[2], feats * channel_mults[1], feats * channel_mults[1])
        self.trans3 = up_conv_block(feats * channel_mults[1], feats * channel_mults[0])
        self.dc3 = conv_block(feats * channel_mults[1], feats * channel_mults[0], feats * 2)
        self.final = nn.Conv3d(feats * 2, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, batch_positions=None):

        # x shape: bxtxcxhxw & out shape: bxcxtxhxw
        out = x.permute(0, 2, 1, 3, 4)

        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=1)  # BxT pad mask
            if self.zero_pad:
                out[out == self.pad_value] = 0

        en3 = self.en3(out)
        pool_3 = self.pool_3(en3)
        en4 = self.en4(pool_3)
        pool_4 = self.pool_4(en4)
        en5 = self.en5(pool_4)
        pool_5 = self.pool_5(en5)

        center_in = self.center_in(pool_5)
        center_out = self.center_out(center_in)

        concat5 = torch.cat([center_out, en5[:, :, :center_out.shape[2], :, :]], dim=1)
        dc5 = self.dc5(concat5)
        trans4 = self.trans4(dc5)
        concat4 = torch.cat([trans4, en4[:, :, :trans4.shape[2], :, :]], dim=1)
        dc4 = self.dc4(concat4)
        trans3 = self.trans3(dc4)
        concat3 = torch.cat([trans3, en3[:, :, :trans3.shape[2], :, :]], dim=1)
        dc3 = self.dc3(concat3)
        final = self.final(dc3)
        final = final.permute(0, 1, 3, 4, 2)  # BxCxHxWxT

        # shape_num = final.shape[0:4]
        # final = final.reshape(-1,final.shape[4])
        if self.pad_value is not None:
            if pad_mask.any():
                # masked mean
                pad_mask = pad_mask[:, :final.shape[-1]] #match new temporal length (due to pooling)
                pad_mask = ~pad_mask # 0 on padded values
                out = (final.permute(1, 2, 3, 0, 4) * pad_mask[None, None, None, :, :]).sum(dim=-1) / pad_mask.sum(
                    dim=-1)[None, None, None, :]
                out = out.permute(3, 0, 1, 2)
            else:
                out = final.mean(dim=-1)
        else:
            out = final.mean(dim=-1)

        return out
