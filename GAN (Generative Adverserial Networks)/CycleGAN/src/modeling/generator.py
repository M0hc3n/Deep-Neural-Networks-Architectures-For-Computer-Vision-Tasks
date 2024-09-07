import torch.nn as nn

from modeling.residual_block import ResidualBlock

class Generator(nn.Module):
    def __init__(self, input_shape, res_blocks_num):
        super(Generator, self).__init__()

        channels = input_shape[0]
        out_channels = 64

        # c7s1
        model = self._get_input_output_conv(channels, out_channels)

        in_channels = out_channels

        # downsampling: d128, d256
        for _ in range(2):
            out_channels *= 2  # 128 then 256

            model += self._get_ud_conv(in_channels, out_channels)

            in_channels = out_channels

        # resnet blocks
        for _ in range(res_blocks_num):
            model += [ResidualBlock(in_channels)]  # fixed channels (256)

        # upsampling: u128, u64
        for _ in range(2):
            out_channels //= 2 # 128 then 64

            model += self._get_ud_conv(in_channels, out_channels, is_u=True)

            in_channels = out_channels

        # Output Layer:
        model += self._get_input_output_conv(channels, out_channels, is_output=True)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def _get_ud_conv(self, in_channels, out_channels, is_u=False):
        res = []

        if is_u:
            res = [
                nn.Upsample(scale_factor=2),
            ]

        return res + [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

    def _get_input_output_conv(self, in_channels, out_channels, is_output=False):
        res = [
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 7),
        ]

        if is_output:
            return res + [
                nn.Tanh(),
            ]
        else:
            return res + [
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
