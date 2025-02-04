from torch import nn, cat

from modeling.conv_block import ConvBlock


class InceptionB(nn.Module):
    def __init__(
        self,
        in_channels,
        ch1x1,
        ch7x7red,
        ch7x7,
        ch7x7dbl_red,
        dch7x7dbl,
        pool_proj,
        conv_block=None,
        stride_num=1,
        pool_type="max",
    ):
        super(InceptionB, self).__init__()

        if ch1x1 == 0:
            self.branch1 = None
        else:
            self.branch1 = ConvBlock(
                in_channels, ch1x1, kernel_size=1, stride=1, padding=0
            )

        if stride_num == 1:
            self.branch2 = nn.Sequential(
                ConvBlock(in_channels, ch7x7red, kernel_size=1),
                ConvBlock(ch7x7red, ch7x7, kernel_size=(1, 7), padding=(0, 3)),
                ConvBlock(ch7x7, ch7x7, kernel_size=(7, 1), padding=(3, 0)),
                ConvBlock(ch7x7, ch7x7, kernel_size=3, stride=2),
            )

            self.branch3 = nn.Sequential(
                ConvBlock(in_channels, ch7x7dbl_red, kernel_size=1),
                ConvBlock(ch7x7dbl_red, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                ConvBlock(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                ConvBlock(dch7x7dbl, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                ConvBlock(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                ConvBlock(dch7x7dbl, dch7x7dbl, kernel_size=3, stride=2),
            )

        else:
            self.branch2 = nn.Sequential(
                ConvBlock(in_channels, ch7x7red, kernel_size=1),
                ConvBlock(ch7x7red, ch7x7, kernel_size=(1, 7), padding=(0, 3)),
                ConvBlock(ch7x7, ch7x7, kernel_size=(7, 1), padding=(3, 0)),
            )

            self.branch3 = nn.Sequential(
                ConvBlock(in_channels, ch7x7dbl_red, kernel_size=1),
                ConvBlock(ch7x7dbl_red, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                ConvBlock(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                ConvBlock(dch7x7dbl, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                ConvBlock(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
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
