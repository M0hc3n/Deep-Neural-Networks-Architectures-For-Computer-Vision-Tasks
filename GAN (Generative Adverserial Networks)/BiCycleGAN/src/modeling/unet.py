import torch

from torch import nn


class UnetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super(UnetDown, self).__init__()

        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]

        if normalize:
            layers += [nn.BatchNorm2d(out_size, 0.8)]

        layers += [nn.LeakyReLU(0.2)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)

        # recall that the UnetUp receives input from preceeding
        # layer and symmetric skip connections corresponding to it
        x = torch.cat((x, skip_input), 1)

        return x
