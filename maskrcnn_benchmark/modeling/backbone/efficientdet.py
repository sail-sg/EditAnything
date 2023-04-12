import torch
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
import cv2
import math
import itertools
import collections
from torchvision.ops import nms


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# https://stackoverflow.com/a/18348004
# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

# in the old version, g_simple_padding = False, which tries to align
# tensorflow's implementation, which is not required here.
g_simple_padding = True
class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        if g_simple_padding:
            self.pool = nn.MaxPool2d(kernel_size, stride,
                                     padding=(kernel_size-1)//2)
        else:
            assert ValueError()
            self.pool = nn.MaxPool2d(kernel_size, stride)
            self.stride = self.pool.stride
            self.kernel_size = self.pool.kernel_size

            if isinstance(self.stride, int):
                self.stride = [self.stride] * 2
            elif len(self.stride) == 1:
                self.stride = [self.stride[0]] * 2

            if isinstance(self.kernel_size, int):
                self.kernel_size = [self.kernel_size] * 2
            elif len(self.kernel_size) == 1:
                self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        if g_simple_padding:
            return self.pool(x)
        else:
            assert ValueError()
            h, w = x.shape[-2:]

            h_step = math.ceil(w / self.stride[1])
            v_step = math.ceil(h / self.stride[0])
            h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
            v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

            extra_h = h_cover_len - w
            extra_v = v_cover_len - h

            left = extra_h // 2
            right = extra_h - left
            top = extra_v // 2
            bottom = extra_v - top

            x = F.pad(x, [left, right, top, bottom])

            x = self.pool(x)
        return x

class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        if g_simple_padding:
            assert kernel_size % 2 == 1
            assert dilation == 1
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                  bias=bias,
                                  groups=groups,
                                  padding=(kernel_size - 1) // 2)
            self.stride = self.conv.stride
            if isinstance(self.stride, int):
                self.stride = [self.stride] * 2
            elif len(self.stride) == 1:
                self.stride = [self.stride[0]] * 2
            else:
                self.stride = list(self.stride)
        else:
            assert ValueError()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                  bias=bias, groups=groups)
            self.stride = self.conv.stride
            self.kernel_size = self.conv.kernel_size
            self.dilation = self.conv.dilation

            if isinstance(self.stride, int):
                self.stride = [self.stride] * 2
            elif len(self.stride) == 1:
                self.stride = [self.stride[0]] * 2

            if isinstance(self.kernel_size, int):
                self.kernel_size = [self.kernel_size] * 2
            elif len(self.kernel_size) == 1:
                self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        if g_simple_padding:
            return self.conv(x)
        else:
            assert ValueError()
            h, w = x.shape[-2:]

            h_step = math.ceil(w / self.stride[1])
            v_step = math.ceil(h / self.stride[0])
            h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
            v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

            extra_h = h_cover_len - w
            extra_v = v_cover_len - h

            left = extra_h // 2
            right = extra_h - left
            top = extra_v // 2
            bottom = extra_v - top

            x = F.pad(x, [left, right, top, bottom])

            x = self.conv(x)
            return x

class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False,
                 epsilon=1e-4, onnx_export=False, attention=True,
                 adaptive_up=False):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.adaptive_up = adaptive_up

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            if len(conv_channels) == 3:
                self.p5_to_p6 = nn.Sequential(
                    Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                    nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                    MaxPool2dStaticSamePadding(3, 2)
                )
            else:
                assert len(conv_channels) == 4
                self.p6_down_channel = nn.Sequential(
                    Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                    nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                )

            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            if len(inputs) == 3:
                p3, p4, p5 = inputs
                p6_in = self.p5_to_p6(p5)
            else:
                p3, p4, p5, p6 = inputs
                p6_in = self.p6_down_channel(p6)

            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        if not self.adaptive_up:
            # Weights for P6_0 and P7_0 to P6_1
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # Weights for P5_0 and P6_0 to P5_1
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            # Connections for P5_0 and P6_0 to P5_1 respectively
            p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

            # Weights for P4_0 and P5_0 to P4_1
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            # Connections for P4_0 and P5_0 to P4_1 respectively
            p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

            # Weights for P3_0 and P4_1 to P3_2
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            # Connections for P3_0 and P4_1 to P3_2 respectively
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))
        else:
            # Weights for P6_0 and P7_0 to P6_1
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_upsample = nn.Upsample(size=p6_in.shape[-2:])
            p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * p6_upsample(p7_in)))

            # Weights for P5_0 and P6_0 to P5_1
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            # Connections for P5_0 and P6_0 to P5_1 respectively
            p5_upsample = nn.Upsample(size=p5_in.shape[-2:])
            p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * p5_upsample(p6_up)))

            # Weights for P4_0 and P5_0 to P4_1
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            # Connections for P4_0 and P5_0 to P4_1 respectively
            p4_upsample = nn.Upsample(size=p4_in.shape[-2:])
            p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * p4_upsample(p5_up)))

            # Weights for P3_0 and P4_1 to P3_2
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_upsample = nn.Upsample(size=p3_in.shape[-2:])
            # Connections for P3_0 and P4_1 to P3_2 respectively
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)
            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        if torch._C._get_tracing_state():
            return x * torch.sigmoid(x)
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers,
                 onnx_export=False, prior_prob=0.01):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)

        prior_prob = prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.normal_(self.header.pointwise_conv.conv.weight, std=0.01)
        torch.nn.init.constant_(self.header.pointwise_conv.conv.bias, bias_value)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        #feats = feats.sigmoid()

        return feats

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        raise ValueError('tend to be deprecated')
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

#TODO: it seems like the standard conv layer is good enough with proper padding
# parameters.
def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        raise ValueError('not validated')
        return Conv2dDynamicSamePadding
    else:
        from functools import partial
        return partial(Conv2dStaticSamePadding, image_size=image_size)

def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        if isinstance(s, (tuple, list)) and all([s0 == s[0] for s0 in s]):
            s = s[0]
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings

def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params

url_map = {
    'efficientnet-b0': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://publicmodels.blob.core.windows.net/container/aa/efficientnet-b7-dcc49843.pth',
}

url_map_advprop = {
    'efficientnet-b0': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b0-b64d5a18.pth',
    'efficientnet-b1': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b1-0f3ce85a.pth',
    'efficientnet-b2': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b2-6e9d97e5.pth',
    'efficientnet-b3': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b3-cdd7c0f4.pth',
    'efficientnet-b4': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b4-44fb3a87.pth',
    'efficientnet-b5': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b5-86493f6b.pth',
    'efficientnet-b6': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b6-ac80338e.pth',
    'efficientnet-b7': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b7-4652b6dd.pth',
    'efficientnet-b8': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b8-22a8fe65.pth',
}

def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    url_map_ = url_map_advprop if advprop else url_map
    from torch.utils import model_zoo
    state_dict = model_zoo.load_url(url_map_[model_name], map_location=torch.device('cpu'))
    # state_dict = torch.load('../../weights/backbone_efficientnetb0.pth')
    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        print(ret)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))

class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, load_weights=True, advprop=True, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        if load_weights:
            load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

class EfficientNetD(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super().__init__()
        model = EfficientNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if tuple(block._depthwise_conv.stride) == (2, 2):
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]

class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.buffer = {}

    @torch.no_grad()
    def forward(self, image, dtype=torch.float32, features=None):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]
        anchor_key = self.get_key('anchor', image_shape)
        stride_idx_key = self.get_key('anchor_stride_index', image_shape)

        if anchor_key in self.buffer:
            return {'stride_idx': self.buffer[stride_idx_key].detach(),
                    'anchor': self.buffer[anchor_key].detach()}

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        all_idx_strides = []
        for idx_stride, stride in enumerate(self.strides):
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if features is not None:
                    f_h, f_w = features[idx_stride].shape[-2:]
                    x = np.arange(stride / 2, stride * f_w, stride)
                    y = np.arange(stride / 2, stride * f_h, stride)
                else:
                    if image_shape[1] % stride != 0:
                        x_max = stride * ((image_shape[1] + stride - 1) // stride)
                        y_max = stride * ((image_shape[0] + stride - 1) // stride)
                    else:
                        x_max = image_shape[1]
                        y_max = image_shape[0]
                    x = np.arange(stride / 2, x_max, stride)
                    y = np.arange(stride / 2, y_max, stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0
                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_level = boxes_level.reshape([-1, 4])
            idx_strides = torch.tensor([idx_stride] * len(boxes_level))
            all_idx_strides.append(idx_strides)
            boxes_all.append(boxes_level)

        anchor_boxes = np.vstack(boxes_all)
        anchor_stride_indices = torch.cat(all_idx_strides).to(image.device)

        self.buffer[stride_idx_key] = anchor_stride_indices

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.buffer[anchor_key] = anchor_boxes

        return {'stride_idx': self.buffer[stride_idx_key],
                'anchor': self.buffer[anchor_key]}

    def get_key(self, hint, image_shape):
        return '{}_{}'.format(hint, '_'.join(map(str, image_shape)))

class EffNetFPN(nn.Module):
    def __init__(self, compound_coef=0, start_from=3):
        super().__init__()

        self.backbone_net = EfficientNetD(EfficientDetBackbone.backbone_compound_coef[compound_coef],
                                          load_weights=False)
        if start_from == 3:
            conv_channel_coef = EfficientDetBackbone.conv_channel_coef[compound_coef]
        else:
            conv_channel_coef = EfficientDetBackbone.conv_channel_coef2345[compound_coef]
        self.bifpn = nn.Sequential(
            *[BiFPN(EfficientDetBackbone.fpn_num_filters[compound_coef],
                    conv_channel_coef,
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    adaptive_up=True)
              for _ in range(EfficientDetBackbone.fpn_cell_repeats[compound_coef])])

        self.out_channels = EfficientDetBackbone.fpn_num_filters[compound_coef]

        self.start_from = start_from
        assert self.start_from in [2, 3]

    def forward(self, inputs):
        if self.start_from == 3:
            _, p3, p4, p5 = self.backbone_net(inputs)

            features = (p3, p4, p5)
            features = self.bifpn(features)
            return features
        else:
            p2, p3, p4, p5 = self.backbone_net(inputs)
            features = (p2, p3, p4, p5)
            features = self.bifpn(features)
            return features

class EfficientDetBackbone(nn.Module):
    backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
    fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
    conv_channel_coef = {
        # the channels of P3/P4/P5.
        0: [40, 112, 320],
        1: [40, 112, 320],
        2: [48, 120, 352],
        3: [48, 136, 384],
        4: [56, 160, 448],
        5: [64, 176, 512],
        6: [72, 200, 576],
        7: [72, 200, 576],
    }
    conv_channel_coef2345 = {
        # the channels of P2/P3/P4/P5.
        0: [24, 40, 112, 320],
        # to be determined for the following
        1: [24, 40, 112, 320],
        2: [24, 48, 120, 352],
        3: [32, 48, 136, 384],
        4: [32, 56, 160, 448],
        5: [40, 64, 176, 512],
        6: [72, 200],
        7: [72, 200],
    }
    fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False,
                 prior_prob=0.01, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    self.conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    adaptive_up=kwargs.get('adaptive_up'))
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     prior_prob=prior_prob)
        anchor_scale = self.anchor_scale[compound_coef]
        if kwargs.get('anchor_scale'):
            anchor_scale = kwargs.pop('anchor_scale')
        if 'anchor_scale' in kwargs:
            del kwargs['anchor_scale']
        self.anchors = Anchors(anchor_scale=anchor_scale, **kwargs)

        self.backbone_net = EfficientNetD(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype, features=features)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                module.bias.data.zero_()

def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.
        if len(anchors.shape) == 3:
            return torch.stack([xmin, ymin, xmax, ymax], dim=2)
        else:
            return torch.stack([xmin, ymin, xmax, ymax], dim=1)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes

def postprocess2(x, anchors, regression, classification,
                 transformed_anchors, threshold, iou_threshold, max_box):
    anchors = anchors['anchor']
    all_above_th = classification > threshold
    out = []
    num_image = x.shape[0]
    num_class = classification.shape[-1]

    #classification = classification.cpu()
    #transformed_anchors = transformed_anchors.cpu()
    #all_above_th = all_above_th.cpu()
    max_box_pre_nms = 1000
    for i in range(num_image):
        all_rois = []
        all_class_ids = []
        all_scores = []
        for c in range(num_class):
            above_th = all_above_th[i, :, c].nonzero()
            if len(above_th) == 0:
                continue
            above_prob = classification[i, above_th, c].squeeze(1)
            if len(above_th) > max_box_pre_nms:
                _, idx = above_prob.topk(max_box_pre_nms)
                above_th = above_th[idx]
                above_prob = above_prob[idx]
            transformed_anchors_per = transformed_anchors[i,above_th,:].squeeze(dim=1)
            from torchvision.ops import nms
            nms_idx = nms(transformed_anchors_per, above_prob, iou_threshold=iou_threshold)
            if len(nms_idx) > 0:
                all_rois.append(transformed_anchors_per[nms_idx])
                ids = torch.tensor([c] * len(nms_idx))
                all_class_ids.append(ids)
                all_scores.append(above_prob[nms_idx])

        if len(all_rois) > 0:
            rois = torch.cat(all_rois)
            class_ids = torch.cat(all_class_ids)
            scores = torch.cat(all_scores)
            if len(scores) > max_box:
                _, idx = torch.topk(scores, max_box)
                rois = rois[idx, :]
                class_ids = class_ids[idx]
                scores = scores[idx]
            out.append({
                'rois': rois,
                'class_ids': class_ids,
                'scores': scores,
            })
        else:
            out.append({
                'rois': [],
                'class_ids': [],
                'scores': [],
            })

    return out

def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    anchors = anchors['anchor']
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': [],
                'class_ids': [],
                'scores': [],
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        from torchvision.ops import nms
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_,
                'class_ids': classes_,
                'scores': scores_,
            })
        else:
            out.append({
                'rois': [],
                'class_ids': [],
                'scores': [],
            })

    return out

def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].detach().cpu().numpy().astype(np.int)
            logging.info((x1, y1, x2, y2))
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            #obj = obj_list[preds[i]['class_ids'][j]]
            #score = float(preds[i]['scores'][j])

            #cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        #(x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #(255, 255, 0), 1)
            #break
        if imshow:
            cv2.imshow('image', imgs[i])
            cv2.waitKey(0)

def calculate_focal_loss2(classification, target_list, alpha, gamma):
    from maskrcnn_benchmark.layers.sigmoid_focal_loss import sigmoid_focal_loss_cuda
    cls_loss = sigmoid_focal_loss_cuda(classification, target_list.int(), gamma, alpha)
    return cls_loss

def calculate_focal_loss(classification, targets, alpha, gamma):
    classification = classification.sigmoid()
    device = classification.device
    alpha_factor = torch.ones_like(targets) * alpha
    alpha_factor = alpha_factor.to(device)

    alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
    focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

    bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

    cls_loss = focal_weight * bce

    zeros = torch.zeros_like(cls_loss)
    zeros = zeros.to(device)
    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)
    return cls_loss.mean()

def calculate_giou(pred, gt):
    ax1, ay1, ax2, ay2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    bx1, by1, bx2, by2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    a = (ax2 - ax1) * (ay2 - ay1)
    b = (bx2 - bx1) * (by2 - by1)
    max_x1, _ = torch.max(torch.stack([ax1, bx1], dim=1), dim=1)
    max_y1, _ = torch.max(torch.stack([ay1, by1], dim=1), dim=1)
    min_x2, _ = torch.min(torch.stack([ax2, bx2], dim=1), dim=1)
    min_y2, _ = torch.min(torch.stack([ay2, by2], dim=1), dim=1)
    inter = (min_x2 > max_x1) * (min_y2 > max_y1)
    inter = inter * (min_x2 - max_x1) * (min_y2 - max_y1)

    min_x1, _ = torch.min(torch.stack([ax1, bx1], dim=1), dim=1)
    min_y1, _ = torch.min(torch.stack([ay1, by1], dim=1), dim=1)
    max_x2, _ = torch.max(torch.stack([ax2, bx2], dim=1), dim=1)
    max_y2, _ = torch.max(torch.stack([ay2, by2], dim=1), dim=1)
    cover = (max_x2 - min_x1) * (max_y2 - min_y1)
    union = a + b - inter
    iou = inter / (union + 1e-5)
    giou = iou - (cover - union) / (cover + 1e-5)
    return giou

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., cls_loss_type='FL', smooth_bce_pos=0.99,
                 smooth_bce_neg=0.01,
                 reg_loss_type='L1',
                 at_least_1_assgin=False,
                 neg_iou_th=0.4,
                 pos_iou_th=0.5,
                 cls_weight=1.,
                 reg_weight=1.,
                 ):
        super(FocalLoss, self).__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        self.iter = 0
        self.reg_loss_type = reg_loss_type
        self.regressBoxes = BBoxTransform()
        if cls_loss_type == 'FL':
            from qd.layers.loss import FocalLossWithLogitsNegLoss
            self.cls_loss = FocalLossWithLogitsNegLoss(alpha, gamma)
        elif cls_loss_type == 'BCE':
            from qd.qd_pytorch import BCEWithLogitsNegLoss
            self.cls_loss = BCEWithLogitsNegLoss(reduction='sum')
        elif cls_loss_type == 'SmoothBCE':
            from qd.layers.loss import SmoothBCEWithLogitsNegLoss
            self.cls_loss = SmoothBCEWithLogitsNegLoss(
                pos=smooth_bce_pos, neg=smooth_bce_neg)
        elif cls_loss_type == 'SmoothFL':
            from qd.layers.loss import FocalSmoothBCEWithLogitsNegLoss
            self.cls_loss = FocalSmoothBCEWithLogitsNegLoss(
                alpha=alpha, gamma=2.,
                pos=smooth_bce_pos, neg=smooth_bce_neg)
        else:
            raise NotImplementedError(cls_loss_type)
        self.at_least_1_assgin = at_least_1_assgin

        self.gt_total = 0
        self.gt_saved_by_at_least = 0

        self.neg_iou_th = neg_iou_th
        self.pos_iou_th = pos_iou_th

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

        self.buf = {}

    def forward(self, classifications, regressions, anchor_info, annotations, **kwargs):
        debug = (self.iter % 100) == 0
        self.iter += 1
        if debug:
            from collections import defaultdict
            debug_info = defaultdict(list)

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchors = anchor_info['anchor']
        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        #anchor_widths = anchor[:, 2] - anchor[:, 0]
        #anchor_heights = anchor[:, 3] - anchor[:, 1]
        #anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        #anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        device = classifications.device

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            #classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                #cls_loss = calculate_focal_loss2(classification,
                                                 #torch.zeros(len(classification)), alpha,
                                                #gamma)
                #cls_loss = cls_loss.mean()
                cls_loss = torch.tensor(0).to(dtype).to(device)
                regression_losses.append(torch.tensor(0).to(dtype).to(device))
                classification_losses.append(cls_loss)
                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            if self.at_least_1_assgin:
                iou_max_gt, iou_argmax_gt = torch.max(IoU, dim=0)
                curr_saved = (iou_max_gt < self.pos_iou_th).sum()
                self.gt_saved_by_at_least += curr_saved
                self.gt_total += len(iou_argmax_gt)
                IoU_max[iou_argmax_gt] = 1.
                IoU_argmax[iou_argmax_gt] = torch.arange(len(iou_argmax_gt)).to(device)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            targets = targets.to(device)

            targets[torch.lt(IoU_max, self.neg_iou_th), :] = 0

            positive_indices = torch.ge(IoU_max, self.pos_iou_th)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if debug:
                if num_positive_anchors > 0:
                    debug_info['pos_conf'].append(classification[
                        positive_indices,
                        assigned_annotations[positive_indices, 4].long()].mean())
                debug_info['neg_conf'].append(classification[targets == 0].mean())
                stride_idx = anchor_info['stride_idx']
                positive_stride_idx = stride_idx[positive_indices]
                pos_count_each_stride = torch.tensor(
                    [(positive_stride_idx == i).sum() for i in range(5)])
                if 'cum_pos_count_each_stride' not in self.buf:
                    self.buf['cum_pos_count_each_stride'] = pos_count_each_stride
                else:
                    cum_pos_count_each_stride = self.buf['cum_pos_count_each_stride']
                    cum_pos_count_each_stride += pos_count_each_stride
                    self.buf['cum_pos_count_each_stride'] = cum_pos_count_each_stride

            #cls_loss = calculate_focal_loss(classification, targets, alpha,
                                            #gamma)
            cls_loss = self.cls_loss(classification, targets)

            cls_loss = cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0)
            assert cls_loss == cls_loss
            classification_losses.append(cls_loss)

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                if self.reg_loss_type == 'L1':
                    anchor_widths_pi = anchor_widths[positive_indices]
                    anchor_heights_pi = anchor_heights[positive_indices]
                    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                    # efficientdet style
                    gt_widths = torch.clamp(gt_widths, min=1)
                    gt_heights = torch.clamp(gt_heights, min=1)

                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)
                    targets_dh = torch.log(gt_heights / anchor_heights_pi)

                    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                    targets = targets.t()

                    regression_diff = torch.abs(targets - regression[positive_indices, :])

                    regression_loss = torch.where(
                        torch.le(regression_diff, 1.0 / 9.0),
                        0.5 * 9.0 * torch.pow(regression_diff, 2),
                        regression_diff - 0.5 / 9.0
                    ).mean()
                elif self.reg_loss_type == 'GIOU':
                    curr_regression = regression[positive_indices, :]
                    curr_anchors = anchor[positive_indices]
                    curr_pred_xyxy = self.regressBoxes(curr_anchors,
                                                        curr_regression)
                    regression_loss = 1.- calculate_giou(curr_pred_xyxy, assigned_annotations)
                    regression_loss = regression_loss.mean()
                    assert regression_loss == regression_loss
                else:
                    raise NotImplementedError
                regression_losses.append(regression_loss)
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
        if debug:
            if len(debug_info) > 0:
                logging.info('pos = {}; neg = {}, saved_ratio = {}/{}={:.1f}, '
                             'stride_info = {}'
                             .format(
                                 torch.tensor(debug_info['pos_conf']).mean(),
                                 torch.tensor(debug_info['neg_conf']).mean(),
                                 self.gt_saved_by_at_least,
                                 self.gt_total,
                                 1. * self.gt_saved_by_at_least / self.gt_total,
                                 self.buf['cum_pos_count_each_stride'],
                             ))
        return self.cls_weight * torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               self.reg_weight * torch.stack(regression_losses).mean(dim=0, keepdim=True)

class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.criterion = criterion
        self.module = model

    def forward(self, *args):
        if len(args) == 2:
            imgs, annotations = args
        elif len(args) == 1:
            imgs, annotations = args[0][:2]
        _, regression, classification, anchors = self.module(imgs)
        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return {'cls_loss': cls_loss, 'reg_loss': reg_loss}

class TorchVisionNMS(nn.Module):
    def __init__(self, iou_threshold):
        super().__init__()
        self.iou_threshold = iou_threshold

    def forward(self, box, prob):
        nms_idx = nms(box, prob, iou_threshold=self.iou_threshold)
        return nms_idx

class PostProcess(nn.Module):
    def __init__(self, iou_threshold):
        super().__init__()
        self.nms = TorchVisionNMS(iou_threshold)

    def forward(self, x, anchors, regression,
                classification,
                transformed_anchors, threshold, max_box):
        all_above_th = classification > threshold
        out = []
        num_image = x.shape[0]
        num_class = classification.shape[-1]

        #classification = classification.cpu()
        #transformed_anchors = transformed_anchors.cpu()
        #all_above_th = all_above_th.cpu()
        max_box_pre_nms = 1000
        for i in range(num_image):
            all_rois = []
            all_class_ids = []
            all_scores = []
            for c in range(num_class):
                above_th = all_above_th[i, :, c].nonzero()
                if len(above_th) == 0:
                    continue
                above_prob = classification[i, above_th, c].squeeze(1)
                if len(above_th) > max_box_pre_nms:
                    _, idx = above_prob.topk(max_box_pre_nms)
                    above_th = above_th[idx]
                    above_prob = above_prob[idx]
                transformed_anchors_per = transformed_anchors[i,above_th,:].squeeze(dim=1)
                nms_idx = self.nms(transformed_anchors_per, above_prob)
                if len(nms_idx) > 0:
                    all_rois.append(transformed_anchors_per[nms_idx])
                    ids = torch.tensor([c] * len(nms_idx))
                    all_class_ids.append(ids)
                    all_scores.append(above_prob[nms_idx])

            if len(all_rois) > 0:
                rois = torch.cat(all_rois)
                class_ids = torch.cat(all_class_ids)
                scores = torch.cat(all_scores)
                if len(scores) > max_box:
                    _, idx = torch.topk(scores, max_box)
                    rois = rois[idx, :]
                    class_ids = class_ids[idx]
                    scores = scores[idx]
                out.append({
                    'rois': rois,
                    'class_ids': class_ids,
                    'scores': scores,
                })
            else:
                out.append({
                    'rois': [],
                    'class_ids': [],
                    'scores': [],
                })

        return out

class InferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold = 0.01
        self.nms_threshold = 0.5
        self.max_box = 100
        self.debug = False
        self.post_process = PostProcess(self.nms_threshold)

    def forward(self, sample):
        features, regression, classification, anchor_info = self.module(sample['image'])
        anchors = anchor_info['anchor']
        classification = classification.sigmoid()
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, sample['image'])

        preds = self.post_process(sample['image'], anchors, regression,
                            classification, transformed_anchors,
                            self.threshold, self.max_box)

        if self.debug:
            logging.info('debugging')
            imgs = sample['image']
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(preds, imgs, list(map(str, range(80))))

        for p, s in zip(preds, sample['scale']):
            if len(p['rois']) > 0:
                p['rois'] /= s
        return preds

