"""
models/regdino.py imports `encoder` from this module
(`from models.backbones.layers import encoder`), but this file does not
exist in the public FMIR GitHub repository (models/backbones/ is entirely
missing upstream).

*** THIS IS A BEST-EFFORT RECONSTRUCTION, NOT THE AUTHORS' ORIGINAL CODE. ***

The call site is:
    encoder(257, N_s, 3, 1, 1)
i.e. positional args (in_channels, out_channels, kernel_size, stride, padding).
This exact signature/behavior (plain Conv-Nd + LeakyReLU block, no
normalization by default) matches the `encoder` block used across the
LKU-Net / RDP / Fourier-Net family of registration repos, which this
codebase's model zoo (encoderOnlyComplex, VxmLKUnetComplex, RDP, LKUNet, ...
all present in models/__init__.py) is clearly built on top of. It is a
reasonable, low-risk reconstruction, but the exact block used in the FMIR
paper's checkpoints (e.g. normalization choice, activation slope) is not
verifiable without the original file — if your results don't match the
paper, this is one of the first places to double-check.
"""

import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, batchnorm=False):
        super(encoder, self).__init__()

        if batchnorm:
            self.layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2))
        else:
            self.layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=padding, bias=bias),
                nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.layer(x)
