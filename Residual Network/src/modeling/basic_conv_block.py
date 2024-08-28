from torch import nn

from collections import OrderedDict

from modeling.lambda_layer import LambdaLayer

import torch.nn.functional as F


# this is a class that would be used as the building block for the ResidualNet
# it basically implements the convolutional block depending on the given option
#       - Option A: increase channel by using identity shortcuts (with null padding)
#       - Option B: increase channel using 1*1 convolution (projection shortcut)
class BasicConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, option="A"):
        super(BasicConvBlock, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", self._get_conv(input_channels, output_channels, stride)),
                    ("bn1", nn.BatchNorm2d(output_channels)),
                    ("act1", nn.ReLU()),
                    (
                        "conv2",
                        self._get_conv(output_channels, output_channels, stride=1),
                    ),
                    ("bn2", nn.BatchNorm2d(output_channels)),
                ],
            )
        )
        
        self.shortcut = nn.Sequential()

        # only apply shortcut if there is a channel mismatch or stride is larger than 1
        if stride != 1 or input_channels != output_channels:            
            self._option_mapper(
                option=option,
                input_channels=input_channels,
                output_channels=output_channels,
                stride=stride,
            )

    def _get_conv(
        self,
        input_channels,
        output_channels,
        stride,
        kernel_size=3,
        padding=1,
        bias=False,
    ):
        return (
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        )

    def _option_mapper(self, option, input_channels, output_channels, stride):
        if option == "A":
            return self._identity_shortcut(out_c=output_channels)
        else:  # assumption is "B"
            return self._projection_shortcut(input_channels, output_channels, stride)

    def _identity_shortcut(self, out_c):
        padd_to_add = out_c // 4

        self.shortcut = LambdaLayer(
            lambda x: F.pad(
                x[:, :, ::2, ::2], (0, 0, 0, 0, padd_to_add, padd_to_add, 0, 0)
            )
        )
        pass

    def _projection_shortcut(self, in_c, out_c, stride):
        self.shortcut = nn.Sequential(
            OrderedDict(
                [
                    (
                        "proj_conv1",
                        self._get_conv(
                            input_channels=in_c,
                            output_channels=out_c * 2,
                            kernel_size=1,
                            stride=stride,
                            padding=0,
                        ),
                    ),
                    (
                        "proj_bn1",
                        nn.BatchNorm2d(2 * out_c),
                    ),
                ]
            )
        )

    def forward(self, x):
        out = self.features(x)
        # sum it with shortcut layer
        out += self.shortcut(x)
        out = F.relu(out)

        return out
