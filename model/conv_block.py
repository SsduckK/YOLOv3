from torch import nn


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.residual = nn.Sequential(
            BasicConv(channel, channel//2, 1, stride=1, padding=0),
            BasicConv(channel//2, channel, 3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x_shortcut = self.shortcut(x)
        x_residual = self.residual(x)

        return x_shortcut + x_residual


class Top_down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, 1, stride=1, padding=0),
            BasicConv(out_channel, out_channel*2, 3, stride=1, padding=0),
            BasicConv(out_channel*2, out_channel, 1, stride=1, padding=0),
            BasicConv(out_channel, out_channel*2, 3, stride=1, padding=1),
            BasicConv(out_channel*2, out_channel, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.conv(x)

