import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape
        self.output_shape = (1, height // 2**4, width // 2**4)

        self.model = nn.Sequential(
            *self._get_discriminator_block(
                channels, out_channels=64, normalize=False
            ),  # C64
            *self._get_discriminator_block(64, out_channels=128),  # 128
            *self._get_discriminator_block(128, out_channels=256),  # C256
            *self._get_discriminator_block(256, out_channels=512),  # C512
            nn.ZeroPad2d(
                (1, 0, 1, 0)
            ),  # this is to keep same Height and Width after applying a kernel of 4*4
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1),
        )

    def _get_discriminator_block(self, in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        ]

        if normalize:
            layers += [nn.InstanceNorm2d(out_channels)]

        layers += [nn.LeakyReLU(0.2, inplace=True)]

        return layers

    def forward(self, img):
        return self.model(img)
