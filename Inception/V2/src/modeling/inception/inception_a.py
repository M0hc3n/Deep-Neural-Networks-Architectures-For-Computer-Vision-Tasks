from torch import nn, cat

from modeling.conv_block import ConvBlock


class InceptionA(nn.Module):
    def __init__(
        self,
        in_channels,
        ch1x1,
        ch3x3red,
        ch3x3,
        ch3x3dbl_red,
        dch3x3dbl,
        pool_proj,
        conv_block=None,
        stride_num=1,
        pool_type="max",
    ):
        super(InceptionA, self).__init__()

        if ch1x1 == 0:
            self.branch1 = None
        else:
            self.branch1 = ConvBlock(in_channels, ch1x1, kernel_size=1, stride=1)

        if stride_num == 1:
            self.branch2 = nn.Sequential(
                ConvBlock(in_channels, ch3x3red, kernel_size=1, stride=1),
                ConvBlock(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1),
            )

            self.branch3 = nn.Sequential(
                ConvBlock(in_channels, ch3x3dbl_red, kernel_size=1, stride=1),
                ConvBlock(ch3x3dbl_red, dch3x3dbl, kernel_size=3, stride=1, padding=1),
                ConvBlock(dch3x3dbl, dch3x3dbl, kernel_size=3, stride=1, padding=1),
            )

        else:
            self.branch2 = nn.Sequential(
                ConvBlock(in_channels, ch3x3red, kernel_size=1, stride=1),
                ConvBlock(ch3x3red, ch3x3, kernel_size=3, stride=2),
            )

            self.branch3 = nn.Sequential(
                ConvBlock(in_channels, ch3x3dbl_red, kernel_size=1, stride=1),
                ConvBlock(ch3x3dbl_red, dch3x3dbl, kernel_size=3, stride=1, padding=1),
                ConvBlock(dch3x3dbl, dch3x3dbl, kernel_size=3, stride=2),
            )

        if pool_proj != 0:
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1),
            )

        else:
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def _forward(self, x):
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        if self.branch1 is not None:
            branch1 = self.branch1(x)
            outputs = [branch1, branch2, branch3, branch4]
        else:
            outputs = [branch2, branch3, branch4]

        return outputs

    def forward(self, x):
        outs = self._forward(x)

        return cat(outs, 1)
