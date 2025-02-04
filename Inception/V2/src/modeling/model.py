import torch
import torch.nn as nn
from torch import optim

from modeling.conv_block import ConvBlock
from modeling.inception.inception_a import InceptionA
from modeling.inception.inception_b import InceptionB
from modeling.inception.inception_c import InceptionC

from modeling.auxilary import Auxiliary


class InceptionV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        aux_logits=True,
        transform_input=False,
        init_weights=True,
        blocks=None,
    ):
        super(InceptionV2, self).__init__()

        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2)
        self.conv2 = ConvBlock(32, 32, kernel_size=3, stride=1)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv4 = ConvBlock(64, 80, kernel_size=3, stride=1)
        self.conv5 = ConvBlock(80, 192, kernel_size=3, stride=2)
        self.conv6 = ConvBlock(192, 288, kernel_size=3, stride=1, padding=1)

        self.inception3a = InceptionA(288, 64, 64, 64, 64, 96, 64, pool_type="avg")
        self.inception3b = InceptionA(288, 64, 64, 96, 64, 96, 32, pool_type="avg")
        self.inception3c = InceptionA(
            288, 0, 128, 320, 64, 160, 0, pool_type="max", stride_num=2
        )

        self.inception5a = InceptionB(768, 192, 96, 160, 96, 160, 256, pool_type="avg")
        self.inception5b = InceptionB(768, 192, 96, 160, 96, 160, 256, pool_type="avg")
        self.inception5c = InceptionB(768, 192, 96, 160, 96, 160, 256, pool_type="avg")
        self.inception5d = InceptionB(768, 192, 96, 160, 96, 160, 256, pool_type="avg")
        self.inception5e = InceptionB(
            768, 0, 128, 192, 128, 320, 0, pool_type="max", stride_num=2
        )

        self.inception2a = InceptionC(
            1280, 256, 128, 160, 128, 240, 224, pool_type="avg"
        )
        self.inception2b = InceptionC(1280, 256, 96, 96, 96, 160, 0, pool_type="max")

        if aux_logits:
            self.aux = Auxiliary(768, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        if init_weights:
            self.initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats

                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            # this performs a standard normalization that adjusts the mean and std
            # why those values in specific ? bcz those reflect the distribution of the ImageNet dataset
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.conv1(x)
        # N x 32 x 149 x 149
        x = self.conv2(x)
        # N x 32 x 147 x 147
        x = self.conv3(x)
        # N x 64 x 147 x 147
        x = self.maxpool(x)
        # N x 64 x 73 x 73
        x = self.conv4(x)
        # N x 80 x 71 x 71
        x = self.conv5(x)
        # N x 192 x 35 x 35
        x = self.conv6(x)
        # N x 288 x 35 x 35

        x = self.inception3a(x)
        # N x 288 x 35 x 35
        x = self.inception3b(x)
        # N x 288 x 35 x 35
        x = self.inception3c(x)
        # N x 768 x 17 x 17

        x = self.inception5a(x)
        # N x 768 x 17 x 17
        x = self.inception5b(x)
        # N x 768 x 17 x 17
        x = self.inception5c(x)
        # N x 768 x 17 x 17
        x = self.inception5d(x)
        # N x 768 x 17 x 17

        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.aux(x)
        else:
            aux = None

        x = self.inception5e(x)
        # N x 1280 x 8 x 8
        x = self.inception2a(x)
        # N x 1280 x 8 x 8
        x = self.inception2b(x)
        # N x 1280 x 8 x 8

        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.dropout(x)
        # N x 2048
        x = self.fc1(x)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)
        return x, aux

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits

        if aux_defined:
            return x, aux
        else:
            return x

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, learning_rate=0.01):
        return optim.Adam(self.parameters(), lr=learning_rate)
