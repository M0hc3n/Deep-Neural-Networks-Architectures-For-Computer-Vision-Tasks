from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(
                1
            ),  # paper's recommendation is to start with a Reflection Padding to remove artifacts effect
            # here is a good illustration of what it does:  https://blog.kakaocdn.net/dn/cbZmXN/btrqmJMsdcf/QvrfKkKe7adxj9jJT4hEXK/img.png
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(
                in_channels
            ),  # why not batch, Instance normalizes per-instance, while batch does it per-batch.
            # Why ? check this: https://stackoverflow.com/questions/45463778/instance-normalisation-vs-batch-normalisation
            nn.ReLU(
                inplace=True
            ),  # Destroys the original input, can slightly improve memory but not to be used when the input is needed in further operations
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)
