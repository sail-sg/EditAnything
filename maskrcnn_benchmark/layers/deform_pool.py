import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform_conv import DeformConv2d

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

class SPPLayer(nn.Module):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4),dim=1)
        return out

class DropBlock(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)

    def reset(self, block_size, keep_prob):
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)

    def calculate_gamma(self, x):
        return  (1-self.keep_prob) * x.shape[-1]**2/ \
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2)

    def forward(self, x):
        if (not self.training or self.keep_prob==1): #set keep_prob=1 to turn off dropblock
            return x
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
        if x.type() == 'torch.cuda.HalfTensor': #TODO: not fully support for FP16 now
            FP16 = True
            x = x.float()
        else:
            FP16 = False
        p = torch.ones_like(x) * (self.gamma)
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)

        out =  mask * x * (mask.numel()/mask.sum())

        if FP16:
            out = out.half()
        return out

class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch//2, 1, 1))
            resblock_one.append(add_conv(ch//2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class RFBblock(nn.Module):
    def __init__(self,in_ch,residual=False):
        super(RFBblock, self).__init__()
        inter_c = in_ch // 4
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=2, padding=2)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=3, padding=3)
        )
        self.residual= residual

    def forward(self,x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        out = torch.cat((x_0,x_1,x_2,x_3),1)
        if self.residual:
            out +=x
        return out


class FeatureAdaption(nn.Module):
    def __init__(self, in_ch, out_ch, n_anchors, rfb=False, sep=False):
        super(FeatureAdaption, self).__init__()
        if sep:
            self.sep=True
        else:
            self.sep=False
            self.conv_offset = nn.Conv2d(in_channels=2*n_anchors,
                                         out_channels=2*9*n_anchors, groups = n_anchors, kernel_size=1,stride=1,padding=0)
            self.dconv = DeformConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,
                                      padding=1, deformable_groups=n_anchors)
            self.rfb=None
            if rfb:
                self.rfb = RFBblock(out_ch)

    def forward(self, input, wh_pred):
        #The RFB block is added behind FeatureAdaption
        #For mobilenet, we currently don't support rfb and FeatureAdaption
        if self.sep:
            return input
        if self.rfb is not None:
            input = self.rfb(input)
        wh_pred_new = wh_pred.detach()
        offset = self.conv_offset(wh_pred_new)
        out = self.dconv(input, offset)
        return out


class ASFFmobile(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFFmobile, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2, leaky=False)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2, leaky=False)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1, leaky=False)
        elif level==1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1, leaky=False)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2, leaky=False)
            self.expand = add_conv(self.inter_dim, 512, 3, 1, leaky=False)
        elif level==2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1, leaky=False)
            self.compress_level_1 = add_conv(256, self.inter_dim, 1, 1, leaky=False)
            self.expand = add_conv(self.inter_dim, 256, 3, 1,leaky=False)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized =F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+ \
                            level_1_resized * levels_weight[:,1:2,:,:]+ \
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+ \
                            level_1_resized * levels_weight[:,1:2,:,:]+ \
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

def add_sepconv(in_ch, out_ch, ksize, stride):

    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('sepconv', nn.Conv2d(in_channels=in_ch,
                                          out_channels=in_ch, kernel_size=ksize, stride=stride,
                                          padding=pad, groups=in_ch, bias=False))
    stage.add_module('sepbn', nn.BatchNorm2d(in_ch))
    stage.add_module('seprelu6', nn.ReLU6(inplace=True))
    stage.add_module('ptconv', nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
    stage.add_module('ptbn', nn.BatchNorm2d(out_ch))
    stage.add_module('ptrelu6', nn.ReLU6(inplace=True))
    return stage

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ressepblock(nn.Module):
    def __init__(self, ch, out_ch, in_ch=None, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        in_ch = ch//2 if in_ch==None else in_ch
        resblock_one = nn.ModuleList()
        resblock_one.append(add_conv(ch, in_ch, 1, 1, leaky=False))
        resblock_one.append(add_conv(in_ch, out_ch, 3, 1,leaky=False))
        self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

