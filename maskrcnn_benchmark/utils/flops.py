import argparse
import logging
import torch
import torch.nn as nn
import timeit

from maskrcnn_benchmark.layers import *
from maskrcnn_benchmark.modeling.backbone.resnet_big import StdConv2d
from maskrcnn_benchmark.modeling.backbone.fpn import *
from maskrcnn_benchmark.modeling.rpn.inference import *
from maskrcnn_benchmark.modeling.roi_heads.box_head.inference import PostProcessor
from maskrcnn_benchmark.modeling.rpn.anchor_generator import BufferList


def profile(model, input_size, custom_ops={}, device="cpu", verbose=False, extra_args={}, return_time=False):
    handler_collection = []

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            print("Not implemented for ", m)

        if fn is not None:
            if verbose:
                print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval().to(device)
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(device)
    with torch.no_grad():
        tic = timeit.time.perf_counter()
        model(x, **extra_args)
        toc = timeit.time.perf_counter()
        total_time = toc-tic

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    if return_time:
        return total_ops, total_params, total_time
    else:
        return total_ops, total_params


multiply_adds = 1
def count_conv2d(m, x, y):
    x = x[0]
    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size()[0]
    out_h = y.size(2)
    out_w = y.size(3)
    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
    kernel_ops = multiply_adds * kh * kw * cin // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops
    # total ops
    # num_out_elements = y.numel()
    output_elements = batch_size * out_w * out_h * cout
    total_ops = output_elements * ops_per_element
    m.total_ops = torch.Tensor([int(total_ops)])


def count_convtranspose2d(m, x, y):
    x = x[0]
    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size()[0]
    out_h = y.size(2)
    out_w = y.size(3)
    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
    kernel_ops = multiply_adds * kh * kw * cin // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops
    # total ops
    # num_out_elements = y.numel()
    # output_elements = batch_size * out_w * out_h * cout
    ops_per_element = m.weight.nelement()
    output_elements = y.nelement()
    total_ops = output_elements * ops_per_element
    m.total_ops = torch.Tensor([int(total_ops)])


def count_bn(m, x, y):
    x = x[0]
    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4*nelements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
    x = x[0]
    batch_size, nfeatures = x.size()
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    m.total_ops = torch.Tensor([int(total_ops)])


def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size]))
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_maxpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])])//torch.Tensor(list((m.output_size,))).squeeze()
    kernel_ops = torch.prod(kernel)
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])])//torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_LastLevelMaxPool(m, x, y):
    num_elements = y[-1].numel()
    total_ops = num_elements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_ROIAlign(m, x, y):
    num_elements = y.numel()
    total_ops = num_elements*4
    m.total_ops = torch.Tensor([int(total_ops)])


register_hooks = {
    Scale: None,
    Conv2d: count_conv2d,
    nn.Conv2d: count_conv2d,
    ModulatedDeformConv: count_conv2d,
    StdConv2d: count_conv2d,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    FrozenBatchNorm2d: count_bn,
    nn.GroupNorm: count_bn,
    NaiveSyncBatchNorm2d: count_bn,

    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    swish: None,

    nn.ConstantPad2d: None,
    SPPLayer: count_LastLevelMaxPool,
    LastLevelMaxPool: count_LastLevelMaxPool,
    nn.MaxPool1d: count_maxpool,
    nn.MaxPool2d: count_maxpool,
    nn.MaxPool3d: count_maxpool,
    nn.AdaptiveMaxPool1d: count_adap_maxpool,
    nn.AdaptiveMaxPool2d: count_adap_maxpool,
    nn.AdaptiveMaxPool3d: count_adap_maxpool,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Upsample: None,
    nn.Dropout: None,
    nn.Sigmoid: None,
    DropBlock2D: None,

    ROIAlign: count_ROIAlign,
    RPNPostProcessor: None,
    PostProcessor: None,
    BufferList: None,
    RetinaPostProcessor: None,
    FCOSPostProcessor: None,
    ATSSPostProcessor: None,
}