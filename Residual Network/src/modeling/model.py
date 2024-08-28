import torch
import torch.nn as nn
from torch import optim

import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks):
        super(ResNet, self).__init__()

        self.in_channels = 16

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        self.block1 = self.__build_layer(
            block_type, 16, num_blocks=num_blocks[0], starting_stride=1
        )
        self.block2 = self.__build_layer(
            block_type, 32, num_blocks=num_blocks[1], starting_stride=2
        )
        self.block3 = self.__build_layer(
            block_type, 64, num_blocks=num_blocks[2], starting_stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 10)

    def __build_layer(self, block_type, out_channels, num_blocks, starting_stride):
        strides_list = [starting_stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides_list:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.linear(out)
        return out

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, learning_rate=0.01):
        return optim.Adam(self.parameters(), lr=learning_rate)
