from modeling.generator import Generator
from modeling.discriminator import Discriminator

import torch.nn as nn

class GAN(nn.Module):
    def __init__(self, noise_shape, data_shape, num_classes):
        super(GAN, self).__init__()

        self.noise_shape = noise_shape
        self.data_shape = data_shape
        self.num_classes = num_classes

        gen_input_dim = self.get_gen_input_dim(noise_shape, num_classes)
        disc_input_dim = self.get_disc_input_dim(data_shape, num_classes)

        self.gen, _, self.gen_opt = Generator().create_model(gen_input_dim, learning_rate=0.0001)
        self.disc, _, self.disc_opt = Discriminator().create_model(disc_input_dim, learning_rate=0.0001)
        
        # self.gen = self.gen.init_weights()
        # self.disc = self.disc.init_weights()
        
        self.criterion = nn.BCEWithLogitsLoss() 

    def get_gen_input_dim(self, noise_shape, num_classes):
        return (
            noise_shape + num_classes
        )  # cuz y is passed to generator in one hot encoding format
        # "+" here means concatenating tensors afterwards

    def get_disc_input_dim(self, data_shape, num_classes):
        return data_shape[0] + num_classes  # same logic goes here
    
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
    
