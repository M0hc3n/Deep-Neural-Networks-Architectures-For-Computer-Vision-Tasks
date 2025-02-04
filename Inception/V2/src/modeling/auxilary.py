from torch import nn, flatten

import torch.nn.functional as F

from modeling.conv_block import ConvBlock


class Auxiliary(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Auxiliary, self).__init__()

        self.conv = ConvBlock(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (5, 5))

        x = self.conv(x)

        x = flatten(x, 1)

        x = F.relu(self.fc1(x), inplace=True)

        x = F.dropout(x, 0.7, training=self.training)

        x = self.fc2(x)
        return x
