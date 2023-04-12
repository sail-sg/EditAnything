from torch import nn

from maskrcnn_benchmark.modeling.make_layers import make_conv3x3


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, use_gn=False):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = make_conv3x3(inp_dim, int(out_dim / 2), 1, use_relu=False, use_gn=use_gn)
        # self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = make_conv3x3(int(out_dim / 2), int(out_dim / 2), 3, use_relu=False, use_gn=use_gn)
        # self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = make_conv3x3(int(out_dim / 2), out_dim, 1, use_relu=False, use_gn=use_gn)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
            self.skip_layer = make_conv3x3(inp_dim, out_dim, 1, use_relu=False, use_gn=False)

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = x
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        # out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, gn=False, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, gn=gn)
        else:
            self.low2 = Residual(nf, nf, gn)
        self.low3 = Residual(nf, f, gn)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2