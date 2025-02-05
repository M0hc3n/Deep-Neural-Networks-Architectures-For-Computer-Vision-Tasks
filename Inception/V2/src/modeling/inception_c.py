from torch import nn, cat

from modeling.conv_block import ConvBlock


class InceptionC(nn.Module):
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
        super(InceptionC, self).__init__()

        self.branch1 = ConvBlock(in_channels, ch1x1, kernel_size=1, stride=1, padding=0)

        self.branch3x3_1 = ConvBlock(in_channels, ch3x3red, kernel_size=1)
        self.branch3x3_2a = ConvBlock(
            ch3x3red, ch3x3, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch3x3_2b = ConvBlock(
            ch3x3red, ch3x3, kernel_size=(3, 1), padding=(1, 0)
        )

        self.branch3x3dbl_1 = ConvBlock(in_channels, ch3x3dbl_red, kernel_size=1)
        self.branch3x3dbl_2 = ConvBlock(
            ch3x3dbl_red, dch3x3dbl, kernel_size=3, padding=1
        )
        self.branch3x3dbl_3a = ConvBlock(
            dch3x3dbl, dch3x3dbl, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch3x3dbl_3b = ConvBlock(
            dch3x3dbl, dch3x3dbl, kernel_size=(3, 1), padding=(1, 0)
        )

        if pool_proj != 0:
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _forward(self, x):
        branch1 = self.branch1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = cat(branch3x3dbl, 1)

        branch4 = self.branch4(x)

        outputs = [branch1, branch3x3, branch3x3dbl, branch4]
        return outputs

    def forward(self, x):
        outs = self._forward(x)

        return cat(outs, 1)
