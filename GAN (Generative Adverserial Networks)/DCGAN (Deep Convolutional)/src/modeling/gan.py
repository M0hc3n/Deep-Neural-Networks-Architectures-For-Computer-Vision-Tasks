import torch.nn as nn
from modeling.generator import Generator
from modeling.discriminator import Discriminator

from core.config import device

import torch.optim as optim

class GAN(nn.Module):
    def __init__(self, learning_rate: float = 0.001):
        super(GAN, self).__init__()

        self.generator, _, self.optimizer_g = Generator().create_model()

        self.discriminator, _, self.optimizer_d = Discriminator().create_model()

        # self.optimizer_g = optim.Adam(
        #     self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        # )
        # self.optimizer_d = optim.Adam(
        #     self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        # )

        self.model = nn.Sequential(self.generator, self.discriminator)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def create_model(self):
        return self, self.criterion
