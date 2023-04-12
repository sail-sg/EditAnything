import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3*dilation, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2*dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def maxpool(**kwargs):
    return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


def avgpool(**kwargs):
    return nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

def dropout(prob):
    return nn.Dropout(prob)


conv3x3sep = lambda i, o, s=1: conv3x3(i, o, s, groups=i)
conv3x3g2 = lambda i, o, s=1: conv3x3(i, o, s, groups=2)
conv3x3g4 = lambda i, o, s=1: conv3x3(i, o, s, groups=4)
conv3x3g8 = lambda i, o, s=1: conv3x3(i, o, s, groups=8)
conv3x3dw = lambda i, o, s=1: conv3x3(i, o, s, groups=i)

conv3x3d2 = lambda i, o, s=1: conv3x3(i, o, s, dilation=2)
conv3x3d3 = lambda i, o, s=1: conv3x3(i, o, s, dilation=3)
conv3x3d4 = lambda i, o, s=1: conv3x3(i, o, s, dilation=4)


conv5x5sep = lambda i, o, s=1: conv5x5(i, o, s, groups=i)
conv5x5g2 = lambda i, o, s=1: conv5x5(i, o, s, groups=2)
conv5x5g4 = lambda i, o, s=1: conv5x5(i, o, s, groups=4)
conv5x5g8 = lambda i, o, s=1: conv5x5(i, o, s, groups=8)
conv5x5dw = lambda i, o, s=1: conv5x5(i, o, s, groups=i)


conv5x5d2 = lambda i, o, s=1: conv5x5(i, o, s, dilation=2)
conv5x5d3 = lambda i, o, s=1: conv5x5(i, o, s, dilation=3)
conv5x5d4 = lambda i, o, s=1: conv5x5(i, o, s, dilation=4)

conv7x7sep = lambda i, o, s=1: conv7x7(i, o, s, groups=i)
conv7x7g2 = lambda i, o, s=1: conv7x7(i, o, s, groups=2)
conv7x7g4 = lambda i, o, s=1: conv7x7(i, o, s, groups=4)
conv7x7g8 = lambda i, o, s=1: conv7x7(i, o, s, groups=8)
conv7x7dw = lambda i, o, s=1: conv7x7(i, o, s, groups=i)

conv7x7d2 = lambda i, o, s=1: conv7x7(i, o, s, dilation=2)
conv7x7d3 = lambda i, o, s=1: conv7x7(i, o, s, dilation=3)
conv7x7d4 = lambda i, o, s=1: conv7x7(i, o, s, dilation=4)