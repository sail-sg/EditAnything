import torch.nn as nn
from .ops import *


class stem(nn.Module):
    num_layer = 1

    def __init__(self, conv, inplanes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(stem, self).__init__()

        self.conv1 = conv(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class basic(nn.Module):
    expansion = 1
    num_layer = 2

    def __init__(self, conv, inplanes, planes, stride=1, midplanes=None, norm_layer=nn.BatchNorm2d):
        super(basic, self).__init__()
        midplanes = planes if midplanes is None else midplanes
        self.conv1 = conv(inplanes, midplanes, stride)
        self.bn1 = norm_layer(midplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(midplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride!=1 or inplanes!=planes*self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class bottleneck(nn.Module):
    expansion = 4
    num_layer = 3

    def __init__(self, conv, inplanes, planes, stride=1, midplanes=None, norm_layer=nn.BatchNorm2d):
        super(bottleneck, self).__init__()
        midplanes = planes if midplanes is None else midplanes
        self.conv1 = conv1x1(inplanes, midplanes)
        self.bn1 = norm_layer(midplanes)
        self.conv2 = conv(midplanes, midplanes, stride)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = conv1x1(midplanes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if stride!=1 or inplanes!=planes*self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes*self.expansion, stride),
                norm_layer(planes*self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class invert(nn.Module):
    def __init__(self, conv, inp, oup, stride=1, expand_ratio=1, norm_layer=nn.BatchNorm2d):
        super(invert, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                conv(hidden_dim, hidden_dim, stride),
                norm_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                norm_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv(hidden_dim, hidden_dim, stride),
                norm_layer(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


invert2 = lambda op, inp, outp, stride, **kwargs: invert(op, inp, outp, stride, expand_ratio=2, **kwargs)
invert3 = lambda op, inp, outp, stride, **kwargs: invert(op, inp, outp, stride, expand_ratio=3, **kwargs)
invert4 = lambda op, inp, outp, stride, **kwargs: invert(op, inp, outp, stride, expand_ratio=4, **kwargs)
invert6 = lambda op, inp, outp, stride, **kwargs: invert(op, inp, outp, stride, expand_ratio=6, **kwargs)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class shuffle(nn.Module):
    expansion = 1
    num_layer = 3

    def __init__(self, conv, inplanes, outplanes, stride=1, midplanes=None, norm_layer=nn.BatchNorm2d):
        super(shuffle, self).__init__()
        inplanes = inplanes // 2 if stride == 1 else inplanes
        midplanes = outplanes // 2 if midplanes is None else midplanes
        rightoutplanes = outplanes - inplanes
        if stride == 2:
            self.left_branch = nn.Sequential(
                # dw
                conv(inplanes, inplanes, stride),
                norm_layer(inplanes),
                # pw-linear
                conv1x1(inplanes, inplanes),
                norm_layer(inplanes),
                nn.ReLU(inplace=True),
            )

        self.right_branch = nn.Sequential(
            # pw
            conv1x1(inplanes, midplanes),
            norm_layer(midplanes),
            nn.ReLU(inplace=True),
            # dw
            conv(midplanes, midplanes, stride),
            norm_layer(midplanes),
            # pw-linear
            conv1x1(midplanes, rightoutplanes),
            norm_layer(rightoutplanes),
            nn.ReLU(inplace=True),
        )

        self.reduce = stride==2

    def forward(self, x):
        if self.reduce:
            out = torch.cat((self.left_branch(x), self.right_branch(x)), 1)
        else:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = torch.cat((x1, self.right_branch(x2)), 1)

        return channel_shuffle(out, 2)


class shufflex(nn.Module):
    expansion = 1
    num_layer = 3

    def __init__(self, conv, inplanes, outplanes, stride=1, midplanes=None, norm_layer=nn.BatchNorm2d):
        super(shufflex, self).__init__()
        inplanes = inplanes // 2 if stride == 1 else inplanes
        midplanes = outplanes // 2 if midplanes is None else midplanes
        rightoutplanes = outplanes - inplanes
        if stride==2:
            self.left_branch = nn.Sequential(
                # dw
                conv(inplanes, inplanes, stride),
                norm_layer(inplanes),
                # pw-linear
                conv1x1(inplanes, inplanes),
                norm_layer(inplanes),
                nn.ReLU(inplace=True),
            )

        self.right_branch = nn.Sequential(
            # dw
            conv(inplanes, inplanes, stride),
            norm_layer(inplanes),
            # pw-linear
            conv1x1(inplanes, midplanes),
            norm_layer(midplanes),
            nn.ReLU(inplace=True),
            # dw
            conv(midplanes, midplanes, 1),
            norm_layer(midplanes),
            # pw-linear
            conv1x1(midplanes, midplanes),
            norm_layer(midplanes),
            nn.ReLU(inplace=True),
            # dw
            conv(midplanes, midplanes, 1),
            norm_layer(midplanes),
            # pw-linear
            conv1x1(midplanes, rightoutplanes),
            norm_layer(rightoutplanes),
            nn.ReLU(inplace=True),
        )

        self.reduce = stride==2

    def forward(self, x):
        if self.reduce:
            out = torch.cat((self.left_branch(x), self.right_branch(x)), 1)
        else:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.right_branch(x2)), 1)

        return channel_shuffle(out, 2)