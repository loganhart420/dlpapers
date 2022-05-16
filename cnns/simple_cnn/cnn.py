from torch import nn

from cnns.layers.layers import ConvBlock


class CNN(nn.Module):
    def __init__(self, channels, img_channels = 3, stride = 2, kernel_size = 3, padding = 0):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(img_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(channels[0], channels[1], stride=2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(channels[1] * 16, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

