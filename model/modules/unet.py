import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from math import sqrt


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# Gama Encoding
# Code is from https://github.com/ivanvovk/WaveGrad/blob/721c37c216132a2ef0a16adc38439f993998e0b7/model/linear_modulation.py
class Conv1dWithInitialization():
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


LINEAR_SCALE = 5000


class PositionalEncoding():
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(
            noise_level) / float(half_dim)
        exponents = 1e-4 ** exponents
        exponents = LINEAR_SCALE * \
            noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


# Resblock Code from BigGAN
# https: // github.com/sxhxliang/BigGAN-pytorch
class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, noise_channel, stride=1, dropout=0, use_affine_level=False):
        super(ResBlockUp, self).__init__()
        self.use_affine_level = use_affine_level
        if self.use_affine_level:
            self.noise_func = Conv1dWithInitialization(
                in_channels=noise_channel,
                out_channels=out_channels*2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        else:
            self.noise_func = Conv1dWithInitialization(
                in_channels=noise_channel,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.model1 = nn.Sequential(
            self.bn1,
            nn.ReLU(),
            nn.Upsample(scale_factor=2) if stride > 1 else nn.Identity(),
            self.conv1,
        )
        self.model2 = nn.Sequential(
            self.bn2,
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            self.conv2
        )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x_in, noise_embed):
        batch = x_in.shape[0]
        x = self.model1(x_in)
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)

        return self.model2(x) + self.bypass(x_in)


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, noise_channel, stride=1, dropout=0, use_affine_level=False):
        super(ResBlockDown, self).__init__()
        self.use_affine_level = use_affine_level
        if self.use_affine_level:
            self.noise_func = Conv1dWithInitialization(
                in_channels=noise_channel,
                out_channels=out_channels*2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        else:
            self.noise_func = Conv1dWithInitialization(
                in_channels=noise_channel,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.model_base1 = nn.Sequential(
            nn.ReLU(),
            self.bn1,
            self.conv1
        )
        self.model_base2 = nn.Sequential(
            nn.ReLU(),
            self.bn2,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            self.conv2
        )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.model_base2 = nn.Sequential(
                self.model_base2,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # add Bypass
            self.bypass_conv = nn.Conv2d(
                in_channels, out_channels, 1, 1, padding=0)
            self.bypass = nn.Sequential(
                self.bn3,
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # Xavier Initialization
            nn.init.xavier_uniform(self.bypass_conv.weight.data, 1.4142)

    def forward(self, x_in, noise_embed):
        batch = x_in.shape[0]
        x = self.model_base1(x_in)
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return self.model_base2(x) + self.bypass(x_in)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        use_affine_level=False,
        image_size=128,
        skip_range=1.0/sqrt(2)
    ):
        super().__init__()
        self.noise_mlp = PositionalEncoding(inner_channel*4)

        self.start_conv = nn.Conv2d(in_channel, inner_channel,
                                    kernel_size=3, padding=1)
        pre_channel = inner_channel
        downs = []
        num_mults = len(channel_mults)
        self.skip_start = int((1-skip_range)*num_mults*res_blocks)

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks-1):
                downs.append(ResBlockUp(
                    pre_channel, channel_mult, use_affine_level=use_affine_level, noise_channel=inner_channel*4, dropout=dropout, stride=1))
                pre_channel = channel_mult
            if not is_last:
                downs.append(ResBlockUp(
                    pre_channel, pre_channel, use_affine_level=use_affine_level, noise_channel=inner_channel*4, dropout=dropout, stride=2))
        self.downs = nn.ModuleList(downs)

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks-1):
                ups.append(ResBlockDown(
                    pre_channel, channel_mult, use_affine_level=use_affine_level, noise_channel=inner_channel*4, dropout=dropout, stride=1))
                pre_channel = channel_mult
            if not is_last:
                ups.append(ResBlockDown(
                    pre_channel, pre_channel, use_affine_level=use_affine_level, noise_channel=inner_channel*4, dropout=dropout, stride=2))

        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, pre_channel),
            Swish(),
            nn.Conv2d(pre_channel, default(
                out_channel, in_channel), 3, padding=1)
        )

    def forward(self, x, noise_level):
        noise_embed = self.noise_mlp(noise_level)
        feats = []
        x = self.start_conv(x)
        for layer in self.downs:
            x = layer(x, noise_embed)
            feats.append(x)

        idx = 0
        for layer in self.ups:
            info = feats.pop()
            if idx >= self.skip_start:
                x = layer(x+info, noise_embed)
            else:
                x = layer(x, noise_embed)
            idx += 1
        return self.final_conv(x)
