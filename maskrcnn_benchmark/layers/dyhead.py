import torch
import torch.nn.functional as F
from torch import nn

from .deform_conv import ModulatedDeformConv
from .dyrelu import h_sigmoid, DYReLU


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 deformable=False,
                 use_gn=False):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        if use_gn:
            self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        else:
            self.bn = None

    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=Conv3x3Norm,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_deform=False
                 ):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu:
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_deform:
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):

            conv_args = dict()
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](x[level - 1], **conv_args))
            if level < len(x) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[level + 1], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False)

            if self.AttnConv is not None:
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(self.relu(mean_fea))

        return next_x


class DyHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(DyHead, self).__init__()
        self.cfg = cfg
        channels    = cfg.MODEL.DYHEAD.CHANNELS
        use_gn      = cfg.MODEL.DYHEAD.USE_GN
        use_dyrelu  = cfg.MODEL.DYHEAD.USE_DYRELU
        use_dyfuse  = cfg.MODEL.DYHEAD.USE_DYFUSE
        use_deform  = cfg.MODEL.DYHEAD.USE_DFCONV

        conv_func = lambda i,o,s : Conv3x3Norm(i,o,s,deformable=use_deform,use_gn=use_gn)

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=conv_func,
                    use_dyrelu=use_dyrelu,
                    use_dyfuse=use_dyfuse,
                    use_deform=use_deform
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

    def forward(self, x):
        dyhead_tower = self.dyhead_tower(x)
        return dyhead_tower