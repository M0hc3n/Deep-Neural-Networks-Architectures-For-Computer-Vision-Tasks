from modeling.generator import Generator
from modeling.discriminator import Discriminator
from modeling.replay_buffer import ReplayBuffer
from modeling.lr_lambda import LRLambda

import torch.nn as nn
import torch.optim as optim

from core.hyperparameters import hp
import itertools


class GAN(nn.Module):
    def __init__(self, input_shape):
        super(GAN, self).__init__()

        self.input_shape = input_shape

        self._initialize_generators(input_shape)

        self._initialize_discriminators(input_shape)

        self._initialize_criterions()

        self._initialize_buffers()

        self._initialize_lr_schedulers()

    def _initialize_generators(self, input_shape):
        # initialize generators
        self.gen_AB = Generator(input_shape, hp.num_residual_blocks).create_model(
            input_shape, hp.num_residual_blocks
        )

        self.gen_BA = Generator(input_shape, hp.num_residual_blocks).create_model(
            input_shape, hp.num_residual_blocks
        )

        # initializing the optimizer of generators
        self.optim_G = optim.Adam(
            itertools.chain(self.gen_AB.parameters(), self.gen_BA.parameters()),
            lr=hp.lr,
            betas=(hp.b1, hp.b2),
        )

        self.gen_AB.apply(self._initialize_conv_weights_normal)
        self.gen_BA.apply(self._initialize_conv_weights_normal)

    def _initialize_discriminators(self, input_shape):
        # initialize discriminators
        self.disc_A, self.optim_disc_A = Discriminator(input_shape).create_model(
            input_shape
        )
        self.disc_B, self.optim_disc_B = Discriminator(input_shape).create_model(
            input_shape
        )

        self.disc_A.apply(self._initialize_conv_weights_normal)
        self.disc_B.apply(self._initialize_conv_weights_normal)

    def _initialize_criterions(self):
        self.criterion_GAN = nn.MSELoss()  # GAN loss
        self.identity_criterion = nn.L1Loss()  # identity loss
        self.cycle_criterion = nn.L1Loss()  # cycle consistency

    def _initialize_buffers(self):
        # initialize buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def _initialize_lr_schedulers(self):
        # initialize learning rate schedulers (the need to use lr_lambda)
        self.lr_scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optim_G,
            lr_lambda=LRLambda(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
        )
        self.lr_scheduler_disc_A = optim.lr_scheduler.LambdaLR(
            self.optim_disc_A,
            lr_lambda=LRLambda(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
        )
        self.lr_scheduler_disc_B = optim.lr_scheduler.LambdaLR(
            self.optim_disc_B,
            lr_lambda=LRLambda(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
        )

    def _initialize_conv_weights_normal(self, m):
        classname = m.__class__.__name__

        if classname.find("Conv") != -1:  # it is a conv layer
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # paper params

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.weight.bias, 0.0)
