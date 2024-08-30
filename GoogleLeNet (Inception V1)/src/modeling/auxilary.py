from torch import nn, flatten


class Auxiliary(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Auxiliary, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()

        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        out = self.act(out)

        out = flatten(out, 1)  # reshaping to one dim tensor

        out = self.fc1(out)
        out = self.act(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out
