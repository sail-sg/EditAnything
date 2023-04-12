import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from maskrcnn_benchmark.utils.amp import custom_fwd, custom_bwd

from maskrcnn_benchmark import _C

class DeformConvFunction(Function):

    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        weight,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64
    ):
        if input is not None and input.dim() != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            DeformConvFunction._output_size(input, weight, ctx.padding,
                                            ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'
            _C.deform_conv_forward(
                input,
                weight,
                offset,
                output,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(3),
                weight.size(2),
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                cur_im2col_step
            )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                _C.deform_conv_backward_input(
                    input,
                    offset,
                    grad_output,
                    grad_input,
                    grad_offset,
                    weight,
                    ctx.bufs_[0],
                    weight.size(3),
                    weight.size(2),
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    ctx.deformable_groups,
                    cur_im2col_step
                )

            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                _C.deform_conv_backward_parameters(
                    input,
                    offset,
                    grad_output,
                    grad_weight,
                    ctx.bufs_[0],
                    ctx.bufs_[1],
                    weight.size(3),
                    weight.size(2),
                    ctx.stride[1],
                    ctx.stride[0],
                    ctx.padding[1],
                    ctx.padding[0],
                    ctx.dilation[1],
                    ctx.dilation[0],
                    ctx.groups,
                    ctx.deformable_groups,
                    1,
                    cur_im2col_step
                )

        return (grad_input, grad_offset, grad_weight, None, None, None, None, None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size

class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        mask,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1
    ):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        _C.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        _C.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            weight.shape[2],
            weight.shape[3],
            ctx.stride,
            ctx.stride,
            ctx.padding,
            ctx.padding,
            ctx.dilation,
            ctx.dilation,
            ctx.groups,
            ctx.deformable_groups,
            ctx.with_bias
        )
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


deform_conv = DeformConvFunction.apply
modulated_deform_conv = ModulatedDeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False
    ):
        assert not bias
        super(DeformConv, self).__init__()
        self.with_bias = bias

        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, offset):
        return deform_conv(input, offset, self.weight, self.stride,
                           self.padding, self.dilation, self.groups,
                           self.deformable_groups)

    def __repr__(self):
        return "".join([
            "{}(".format(self.__class__.__name__),
            "in_channels={}, ".format(self.in_channels),
            "out_channels={}, ".format(self.out_channels),
            "kernel_size={}, ".format(self.kernel_size),
            "stride={}, ".format(self.stride),
            "dilation={}, ".format(self.dilation),
            "padding={}, ".format(self.padding),
            "groups={}, ".format(self.groups),
            "deformable_groups={}, ".format(self.deformable_groups),
            "bias={})".format(self.with_bias),
        ])

class ModulatedDeformConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True
    ):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels,
            in_channels // groups,
            *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input, offset, mask):
        return modulated_deform_conv(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)

    def __repr__(self):
        return "".join([
            "{}(".format(self.__class__.__name__),
            "in_channels={}, ".format(self.in_channels),
            "out_channels={}, ".format(self.out_channels),
            "kernel_size={}, ".format(self.kernel_size),
            "stride={}, ".format(self.stride),
            "dilation={}, ".format(self.dilation),
            "padding={}, ".format(self.padding),
            "groups={}, ".format(self.groups),
            "deformable_groups={}, ".format(self.deformable_groups),
            "bias={})".format(self.with_bias),
        ])

class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConvPack, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, deformable_groups, bias)

        self.conv_offset_mask = nn.Conv2d(
            self.in_channels // self.groups,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)
