#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN

class ResBlock(nn.Module):
    def __init__(self, in_channel=192, out_channel=192, actv='relu', actv2=None, downscale=False, kernel_size=3, device='cuda:0'):
        super().__init__()
        stride = 2 if downscale else 1
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=1, stride=1)
        if actv == 'relu':
            self.actv1 = nn.ReLU(inplace=True)
        elif actv == 'lrelu':
            self.actv1 = nn.LeakyReLU(0.2, inplace=True)

        if actv2 is None:
            self.actv2 = None
        elif actv2 == 'gdn':
            self.actv2 = GDN(out_channel, device)
        elif actv2 == 'igdn':
            self.actv2 = GDN(out_channel, device, inverse=True)
        
        self.downscale = downscale
        if self.downscale:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2)

    def forward(self, x):
        shortcut = x
        if self.downscale:
            shortcut = self.shortcut(shortcut)
        x = self.conv1(x)
        x = self.actv1(x)
        x = self.conv2(x)
        if self.actv2 is not None:
            x = self.actv2(x)
        x = x + shortcut
        return x