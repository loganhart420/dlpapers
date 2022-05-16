from torch.nn import Module
import torch.nn as nn

from cnns.layers.layers import ConvBlock, ShortcutProjection

"""
This implements the residual block described in the paper. It has two 3 by 3 convolution layers.
"""


class ResBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1, kernel_size=3, padding=1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

        self.relu2 = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu1(self.conv1(x))

        x = self.conv2(x)
        return self.relu2(x + shortcut)



class BottleNeckResBlock(Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvBlock(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = ConvBlock(bottleneck_channels, out_channels, kernel_size=1, stride=1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        
        self.relu3 = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu1(self.conv1(x))

        x = self.relu2(self.conv2(x))

        x = self.conv3(x)

        return self.relu3(x + shortcut)
