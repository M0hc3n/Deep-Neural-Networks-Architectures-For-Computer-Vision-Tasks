import torch.nn as nn

import torch.optim as optim

from core.config import device

from core.hyperparameters import hp


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

    def get_discriminator_block(self, in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]

        if normalize:
            layers += [nn.BatchNorm2d(out_channels, 0.8)]

        layers += [nn.LeakyReLU(0.2)]
        
        return layers

    def create_model(self, input_shape):
        model = Discriminator(input_shape).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=hp.lr,
            betas=(hp.b1, hp.b2),
        )

        return model, optimizer
