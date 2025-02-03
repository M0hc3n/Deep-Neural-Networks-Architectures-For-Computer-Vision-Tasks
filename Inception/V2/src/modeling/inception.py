from torch import nn, cat

from modeling.conv_block import ConvBlock

class Inception(nn.Module):
    def __init__(
        self,
        in_channels,
        num1x1,
        num3x3_reduce,
        num3x3,
        num5x5_reduce,
        num5x5,
        pool_proj,
    ):
        super(Inception, self).__init__()

        self.block1 = nn.Sequential(
            ConvBlock(in_channels, num1x1, kernel_size=1, stride=1, padding=0)
        )

        self.block2 = nn.Sequential(
            ConvBlock(in_channels, num3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num3x3_reduce, num3x3, kernel_size=3, stride=1, padding=1),
        )

        self.block3 = nn.Sequential(
            ConvBlock(in_channels, num5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num5x5_reduce, num5x5, kernel_size=5, stride=1, padding=2),
        )

        self.block4 = nn.Sequential(
            nn.MaxPool2d(
                3, stride=1, padding=1, ceil_mode=True
            ),  # adds extra cols (as padding to cover the stride used)
            ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)

        return cat([out1, out2, out3, out4], 1)
