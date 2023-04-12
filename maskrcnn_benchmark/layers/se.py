from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16,
                 use_conv=True, mid_activation=nn.ReLU(inplace=True), out_activation=nn.Sigmoid()):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        mid_channels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True)
        else:
            self.fc1 = nn.Linear(channels, mid_channels)
        self.activ = mid_activation
        if use_conv:
            self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        else:
            self.fc2 = nn.Linear(mid_channels, channels)
        self.sigmoid = out_activation

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x