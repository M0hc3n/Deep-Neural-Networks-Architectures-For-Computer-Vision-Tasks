import torch
import torch.nn as nn
from torch import optim

from modeling.conv_block import ConvBlock
from modeling.inception import Inception
from modeling.auxilary import Auxiliary


class GoogleLeNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogleLeNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception3A = Inception(
            in_channels=192,
            num1x1=64,
            num3x3_reduce=96,
            num3x3=128,
            num5x5_reduce=16,
            num5x5=32,
            pool_proj=32,
        )

        self.inception3B = Inception(
            in_channels=256,
            num1x1=128,
            num3x3_reduce=128,
            num3x3=192,
            num5x5_reduce=32,
            num5x5=96,
            pool_proj=64,
        )

        self.pool3 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception4A = Inception(
            in_channels=480,
            num1x1=192,
            num3x3_reduce=96,
            num3x3=208,
            num5x5_reduce=16,
            num5x5=48,
            pool_proj=64,
        )

        self.inception4B = Inception(
            in_channels=512,
            num1x1=160,
            num3x3_reduce=112,
            num3x3=224,
            num5x5_reduce=24,
            num5x5=64,
            pool_proj=64,
        )

        self.inception4C = Inception(
            in_channels=512,
            num1x1=128,
            num3x3_reduce=128,
            num3x3=256,
            num5x5_reduce=24,
            num5x5=64,
            pool_proj=64,
        )

        self.inception4D = Inception(
            in_channels=512,
            num1x1=112,
            num3x3_reduce=144,
            num3x3=288,
            num5x5_reduce=32,
            num5x5=64,
            pool_proj=64,
        )

        self.inception4E = Inception(
            in_channels=528,
            num1x1=256,
            num3x3_reduce=160,
            num3x3=320,
            num5x5_reduce=32,
            num5x5=128,
            pool_proj=128,
        )

        self.pool4 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception5A = Inception(
            in_channels=832,
            num1x1=256,
            num3x3_reduce=160,
            num3x3=320,
            num5x5_reduce=32,
            num5x5=128,
            pool_proj=128,
        )

        self.inception5B = Inception(
            in_channels=832,
            num1x1=384,
            num3x3_reduce=192,
            num3x3=384,
            num5x5_reduce=48,
            num5x5=128,
            pool_proj=128,
        )

        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self.aux4A = Auxiliary(512, num_classes)
        self.aux4D = Auxiliary(528, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool2(out)

        out = self.inception3A(out)
        out = self.inception3B(out)
        out = self.pool3(out)
        out = self.inception4A(out)

        aux1 = self.aux4A(out)

        out = self.inception4B(out)
        out = self.inception4C(out)
        out = self.inception4D(out)

        aux2 = self.aux4D(out)

        out = self.inception4E(out)
        out = self.pool4(out)
        out = self.inception5A(out)
        out = self.inception5B(out)
        out = self.pool5(out)

        out = torch.flatten(out, 1)  # flattent to one dim tensor
        out = self.dropout(out)
        out = self.fc(out)

        return out, aux1, aux2

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, learning_rate=0.01):
        return optim.Adam(self.parameters(), lr=learning_rate)
