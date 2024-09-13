from torchvision.models import resnet18

from torch import nn, exp

from torch.autograd import Variable

from core.config import Tensor
from core.hyperparameters import hp

import numpy as np


class Encoder:
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.feature_extractor = nn.Sequential(
            *list(resnet18(pretrained=True).children())[:-3]
        )
        # we exclude the last 3 layers
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # we output the characteristics of the distribution: mean and log (var)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.pool(out)

        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)

        reparamterized_out = self.reparameterization(mu, log_var)

        return mu, log_var, reparamterized_out

    def reparameterization(self, mu, log_var):
        std = exp(log_var / 2)

        sampled_z = Variable(
            Tensor(np.random.normal(0, 1, (mu.size(0), hp.latent_dim)))
        )

        return sampled_z * std + mu
