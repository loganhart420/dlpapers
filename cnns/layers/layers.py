from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
        


class ShortcutProjection(nn.Module):
    """
        This is the pytorch implimentation of linear projection with shortcut connection.
        the WsX

        it takes 3 args:
    
            in_channels: is the number of channels in x
    
            out_channels: the number of channels in F(x, {Wi})
    
            stride: the stride length in the convolution operation for F. 
                We do the same stride on the shortcut connection, to match the feature-map size.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
         return self.conv(x)