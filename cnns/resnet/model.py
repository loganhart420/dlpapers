import torch
from torch import nn
from resnet.resblocks import ResBlock, ConvBlock, BottleNeckResBlock


class Resnet(nn.Module):
    """
    params:
        n_blocks: is a list of of number of blocks for each feature map size.

        n_channels: is the number of channels for each feature map size.
            bottlenecks is the number of channels the bottlenecks. If this is None , residual blocks are used.
        
        img_channels: is the number of channels in the input.
        
        first_kernel_size: is the kernel size of the initial convolution layer

    """
    def __init__(self, n_blocks, n_channels, bottlenecks = None, img_channels: int = 3, first_kernel_size: int = 7):
        super().__init__()
        blocks = []

        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        self.conv = ConvBlock(img_channels, n_channels[0], kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)

        prev_channels = n_channels[0]

        for i, channels in enumerate(n_channels):
            stride = 2 if len(blocks) == 0 else 1
            if bottlenecks is None:
                blocks.append(ResBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleNeckResBlock(prev_channels, bottlenecks[i], channels, stride=stride))

            prev_channels = channels

            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResBlock(channels, channels, stride=1))
                else:
                    blocks.append(BottleNeckResBlock(channels, bottlenecks[i], channels, stride=1))
            
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv(x)

        x = self.blocks(x)

        x = x.view(x.shape[0], x.shape[1], -1) # Change x from shape [batch_size, channels, h, w] to [batch_size, channels, h * w]
        print(x.shape)
        return x.mean(dim=1) # global average pooling
