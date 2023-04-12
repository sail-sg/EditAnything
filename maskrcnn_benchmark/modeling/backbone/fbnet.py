"""
FBNet model builder
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, SyncBatchNorm
from maskrcnn_benchmark.layers import Conv2d, interpolate
from maskrcnn_benchmark.layers import NaiveSyncBatchNorm2d, FrozenBatchNorm2d
from maskrcnn_benchmark.layers.misc import _NewEmptyTensorOp


logger = logging.getLogger(__name__)


def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret


class Identity(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Identity, self).__init__()
        self.conv = (
            ConvBNRelu(
                C_in,
                C_out,
                kernel=1,
                stride=stride,
                pad=0,
                no_bias=1,
                use_relu="relu",
                bn_type="bn",
            )
            if C_in != C_out or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
        return out


class CascadeConv3x3(nn.Sequential):
    def __init__(self, C_in, C_out, stride):
        assert stride in [1, 2]
        ops = [
            Conv2d(C_in, C_in, 3, stride, 1, bias=False),
            BatchNorm2d(C_in),
            nn.ReLU(inplace=True),
            Conv2d(C_in, C_out, 3, 1, 1, bias=False),
            BatchNorm2d(C_out),
        ]
        super(CascadeConv3x3, self).__init__(*ops)
        self.res_connect = (stride == 1) and (C_in == C_out)

    def forward(self, x):
        y = super(CascadeConv3x3, self).forward(x)
        if self.res_connect:
            y += x
        return y


class Shift(nn.Module):
    def __init__(self, C, kernel_size, stride, padding):
        super(Shift, self).__init__()
        self.C = C
        kernel = torch.zeros((C, 1, kernel_size, kernel_size), dtype=torch.float32)
        ch_idx = 0

        assert stride in [1, 2]
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = 1

        hks = kernel_size // 2
        ksq = kernel_size ** 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == hks and j == hks:
                    num_ch = C // ksq + C % ksq
                else:
                    num_ch = C // ksq
                kernel[ch_idx : ch_idx + num_ch, 0, i, j] = 1
                ch_idx += num_ch

        self.register_parameter("bias", None)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if x.numel() > 0:
            return nn.functional.conv2d(
                x,
                self.kernel,
                self.bias,
                (self.stride, self.stride),
                (self.padding, self.padding),
                self.dilation,
                self.C,  # groups
            )

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                (self.padding, self.dilation),
                (self.dilation, self.dilation),
                (self.kernel_size, self.kernel_size),
                (self.stride, self.stride),
            )
        ]
        output_shape = [x.shape[0], self.C] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ShiftBlock5x5(nn.Sequential):
    def __init__(self, C_in, C_out, expansion, stride):
        assert stride in [1, 2]
        self.res_connect = (stride == 1) and (C_in == C_out)

        C_mid = _get_divisible_by(C_in * expansion, 8, 8)

        ops = [
            # pw
            Conv2d(C_in, C_mid, 1, 1, 0, bias=False),
            BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            # shift
            Shift(C_mid, 5, stride, 2),
            # pw-linear
            Conv2d(C_mid, C_out, 1, 1, 0, bias=False),
            BatchNorm2d(C_out),
        ]
        super(ShiftBlock5x5, self).__init__(*ops)

    def forward(self, x):
        y = super(ShiftBlock5x5, self).forward(x)
        if self.res_connect:
            y += x
        return y


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class ConvBNRelu(nn.Sequential):
    def __init__(
        self,
        input_depth,
        output_depth,
        kernel,
        stride,
        pad,
        no_bias,
        use_relu,
        bn_type,
        group=1,
        *args,
        **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "nsbn", "sbn", "af", "gn", None]
        assert stride in [1, 2, 4]

        op = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == "sbn":
            bn_op = SyncBatchNorm(output_depth)
        elif bn_type == "nsbn":
            bn_op = NaiveSyncBatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))


class SEModule(nn.Module):
    reduction = 4

    def __init__(self, C):
        super(SEModule, self).__init__()
        mid = max(C // self.reduction, 8)
        conv1 = Conv2d(C, mid, 1, 1, 0)
        conv2 = Conv2d(mid, C, 1, 1, 0)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), conv1, nn.ReLU(inplace=True), conv2, nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.op(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )


def _get_upsample_op(stride):
    assert (
        stride in [1, 2, 4]
        or stride in [-1, -2, -4]
        or (isinstance(stride, tuple) and all(x in [-1, -2, -4] for x in stride))
    )

    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [-x for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode="nearest", align_corners=None)

    return ret, stride


class IRFBlock(nn.Module):
    def __init__(
        self,
        input_depth,
        output_depth,
        expansion,
        stride,
        bn_type="bn",
        kernel=3,
        width_divisor=1,
        shuffle_type=None,
        pw_group=1,
        se=False,
        cdw=False,
        dw_skip_bn=False,
        dw_skip_relu=False,
    ):
        super(IRFBlock, self).__init__()

        assert kernel in [1, 3, 5, 7], kernel

        self.use_res_connect = stride == 1 and input_depth == output_depth
        self.output_depth = output_depth

        mid_depth = int(input_depth * expansion)
        mid_depth = _get_divisible_by(mid_depth, width_divisor, width_divisor)

        # pw
        self.pw = ConvBNRelu(
            input_depth,
            mid_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu="relu",
            bn_type=bn_type,
            group=pw_group,
        )

        # negative stride to do upsampling
        self.upscale, stride = _get_upsample_op(stride)

        # dw
        if kernel == 1:
            self.dw = nn.Sequential()
        elif cdw:
            dw1 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu",
                bn_type=bn_type,
            )
            dw2 = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=1,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )
            self.dw = nn.Sequential(OrderedDict([("dw1", dw1), ("dw2", dw2)]))
        else:
            self.dw = ConvBNRelu(
                mid_depth,
                mid_depth,
                kernel=kernel,
                stride=stride,
                pad=(kernel // 2),
                group=mid_depth,
                no_bias=1,
                use_relu="relu" if not dw_skip_relu else None,
                bn_type=bn_type if not dw_skip_bn else None,
            )

        # pw-linear
        self.pwl = ConvBNRelu(
            mid_depth,
            output_depth,
            kernel=1,
            stride=1,
            pad=0,
            no_bias=1,
            use_relu=None,
            bn_type=bn_type,
            group=pw_group,
        )

        self.shuffle_type = shuffle_type
        if shuffle_type is not None:
            self.shuffle = ChannelShuffle(pw_group)

        self.se4 = SEModule(output_depth) if se else nn.Sequential()

        self.output_depth = output_depth

    def forward(self, x):
        y = self.pw(x)
        if self.shuffle_type == "mid":
            y = self.shuffle(y)
        if self.upscale is not None:
            y = self.upscale(y)
        y = self.dw(y)
        y = self.pwl(y)
        if self.use_res_connect:
            y += x
        y = self.se4(y)
        return y



skip = lambda C_in, C_out, stride, **kwargs: Identity(
    C_in, C_out, stride
)
basic_block = lambda C_in, C_out, stride, **kwargs: CascadeConv3x3(
    C_in, C_out, stride
)
# layer search 2
ir_k3_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, **kwargs
)
ir_k3_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=3, **kwargs
)
ir_k3_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=3, **kwargs
)
ir_k3_s4 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 4, stride, kernel=3, shuffle_type="mid", pw_group=4, **kwargs
)
ir_k5_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=5, **kwargs
)
ir_k5_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=5, **kwargs
)
ir_k5_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=5, **kwargs
)
ir_k5_s4 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 4, stride, kernel=5, shuffle_type="mid", pw_group=4, **kwargs
)
# layer search se
ir_k3_e1_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, se=True, **kwargs
)
ir_k3_e3_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=3, se=True, **kwargs
)
ir_k3_e6_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=3, se=True, **kwargs
)
ir_k3_s4_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    4,
    stride,
    kernel=3,
    shuffle_type=mid,
    pw_group=4,
    se=True,
    **kwargs
)
ir_k5_e1_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=5, se=True, **kwargs
)
ir_k5_e3_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=5, se=True, **kwargs
)
ir_k5_e6_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=5, se=True, **kwargs
)
ir_k5_s4_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    4,
    stride,
    kernel=5,
    shuffle_type="mid",
    pw_group=4,
    se=True,
    **kwargs
)
# layer search 3 (in addition to layer search 2)
ir_k3_s2 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, shuffle_type="mid", pw_group=2, **kwargs
)
ir_k5_s2 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=5, shuffle_type="mid", pw_group=2, **kwargs
)
ir_k3_s2_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    1,
    stride,
    kernel=3,
    shuffle_type="mid",
    pw_group=2,
    se=True,
    **kwargs
)
ir_k5_s2_se = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in,
    C_out,
    1,
    stride,
    kernel=5,
    shuffle_type="mid",
    pw_group=2,
    se=True,
    **kwargs
)
# layer search 4 (in addition to layer search 3)
ir_k33_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=3, cdw=True, **kwargs
)
ir_k33_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=3, cdw=True, **kwargs
)
ir_k33_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=3, cdw=True, **kwargs
)
# layer search 5 (in addition to layer search 4)
ir_k7_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=7, **kwargs
)
ir_k7_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=7, **kwargs
)
ir_k7_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=7, **kwargs
)
ir_k7_sep_e1 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 1, stride, kernel=7, cdw=True, **kwargs
)
ir_k7_sep_e3 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 3, stride, kernel=7, cdw=True, **kwargs
)
ir_k7_sep_e6 = lambda C_in, C_out, stride, **kwargs: IRFBlock(
    C_in, C_out, 6, stride, kernel=7, cdw=True, **kwargs
)