from modeling.generator import Generator
from modeling.discriminator import Discriminator
from modeling.encoder import Encoder

import torch.nn as nn

from core.hyperparameters import hp


class GAN(nn.Module):
    def __init__(self, input_shape):
        super(GAN, self).__init__()

        self.input_shape = input_shape

        self._initialize_generator(input_shape)

        self._initialize_encoder()

        self._initialize_discriminators(input_shape)

        self._initialize_criterions()

    def _initialize_generator(self, input_shape):
        # initialize generator
        self.gen, self.optim_G = Generator(hp.latent_dim, input_shape).create_model()

        self.gen.apply(self._initialize_conv_weights_normal)

    def _initialize_encoder(self):
        self.enc, self.optim_E = Encoder(hp.latent_dim).create_model()

    def _initialize_discriminators(self, input_shape):
        # initialize discriminators
        self.D_VAE, self.optim_D_VAE = Discriminator(input_shape).create_model()
        self.D_LR, self.optim_D_LR = Discriminator(input_shape).create_model()

        self.D_VAE.apply(self._initialize_conv_weights_normal)
        self.D_LR.apply(self._initialize_conv_weights_normal)

    def _initialize_criterions(self):
        self.mae_loss = nn.L1Loss()

    # def _initialize_lr_schedulers(self):
    #     # initialize learning rate schedulers (the need to use lr_lambda)
    #     self.lr_scheduler_G = optim.lr_scheduler.LambdaLR(
    #         self.optim_G,
    #         lr_lambda=LRLambda(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
    #     )
    #     self.lr_scheduler_disc_A = optim.lr_scheduler.LambdaLR(
    #         self.optim_disc_A,
    #         lr_lambda=LRLambda(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
    #     )
    #     self.lr_scheduler_disc_B = optim.lr_scheduler.LambdaLR(
    #         self.optim_disc_B,
    #         lr_lambda=LRLambda(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
    #     )

    def _initialize_conv_weights_normal(self, m):
        classname = m.__class__.__name__

        if classname.find("Conv") != -1:  # it is a conv layer
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # paper params

        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.weight.bias, 0.0)
