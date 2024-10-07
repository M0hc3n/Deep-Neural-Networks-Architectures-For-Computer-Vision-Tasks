import torch.nn as nn
import torch.optim as optim
from torch import mean

from core.config import device

from core.hyperparameters import hp


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.channels, self.h, self.w = input_shape

        self.models = nn.ModuleList()

        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *self.get_discriminator_block(self.channels, 64, normalize=False),
                    *self.get_discriminator_block(64, 128),
                    *self.get_discriminator_block(128, 256),
                    *self.get_discriminator_block(256, 512),
                    nn.Conv2d(512, 1, kernel_size=3, padding=1),
                ),
            )

        self.downsample = nn.AvgPool2d(
            self.channels, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(self, x):
        outputs = []

        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)

        return outputs

    def compute_loss(self, x, x_true):
        outputs = self.forward(x)

        # we use L2 loss
        loss = sum([mean((out - x_true) ** 2) for out in outputs])

        return loss

    def get_discriminator_block(self, in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]

        if normalize:
            layers += [nn.BatchNorm2d(out_channels, 0.8)]

        layers += [nn.LeakyReLU(0.2)]

        return layers

    def create_model(self):
        model = Discriminator((self.channels, self.h, self.w)).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=hp.lr,
            betas=(hp.b1, hp.b2),
        )

        return model, optimizer
