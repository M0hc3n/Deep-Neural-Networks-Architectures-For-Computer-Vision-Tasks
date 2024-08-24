from modeling.generator import Generator
from modeling.discriminator import Discriminator

from core.config import device

import torch.nn as nn
from torch import optim

class GAN(nn.Module):
    def __init__(self, random_shape, data_shape, num_classes):
        super(GAN, self).__init__()

        self.random_shape = random_shape
        self.data_shape = data_shape
        self.num_classes = num_classes

        gen_input_dim = self.get_gen_input_dim(random_shape, num_classes)
        disc_input_dim = self.get_disc_input_dim(data_shape, num_classes)

        self.gen = Generator(noise_shape=gen_input_dim).to(device)
        self.disc = Discriminator(noise_shape=disc_input_dim).to(device)
        
        self.gen = self.gen.apply(self.init_weights)
        self.disc = self.disc.apply(self.init_weights)
        
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=0.0001)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=0.0001)
        
        self.criterion = nn.BCEWithLogitsLoss() 

    def get_gen_input_dim(self, random_shape, num_classes):
        return (
            random_shape + num_classes
        )  # cuz y is passed to generator in one hot encoding format
        # "+" here means concatenating tensors afterwards

    def get_disc_input_dim(self, data_shape, num_classes):
        return data_shape + num_classes  # same logic goes here
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

    def create_model(self):
        return self, self.criterion
    
