# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None, drop_block=None, use_spp=False, use_pan=False,
            return_swint_feature_before_fusion=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.pan_blocks = [] if use_pan else None
        self.spp_block = SPPLayer() if use_spp else None
        self.return_swint_feature_before_fusion = return_swint_feature_before_fusion
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            if idx==len(in_channels_list) and use_spp:
                in_channels = in_channels*4
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

            if use_pan:
                pan_in_block = "pan_in_layer{}".format(idx)
                pan_in_block_module = conv_block(out_channels, out_channels, 3, 2)
                self.add_module(pan_in_block, pan_in_block_module)
                pan_out_block = "pan_out_layer{}".format(idx)
                pan_out_block_module = conv_block(out_channels, out_channels, 3, 1)
                self.add_module(pan_out_block, pan_out_block_module)
                self.pan_blocks.append([pan_in_block, pan_out_block])

        self.top_blocks = top_blocks
        self.drop_block = drop_block

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        if type(x) is tuple:
            # for the case of VL backbone
            x, x_text = x[0], x[1]
        # print([v.shape for v in x])
        swint_feature_c4 = None
        if self.return_swint_feature_before_fusion:
            # TODO: here we only return last single scale feature map before the backbone fusion, should be more flexible
            swint_feature_c4 = x[-2]

        if self.spp_block:
            last_inner = getattr(self, self.inner_blocks[-1])(self.spp_block(x[-1]))
        else:
            last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)

            if inner_lateral.shape[-2:] != last_inner.shape[-2:]:
                # TODO: could also give size instead of
                inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode="nearest")
            else:
                inner_top_down = last_inner

            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            if self.drop_block and self.training:
                results.insert(0, getattr(self, layer_block)(self.drop_block(last_inner)))
            else:
                results.insert(0, getattr(self, layer_block)(last_inner))

        if self.pan_blocks:
            pan_results = []
            last_outer = results[0]
            pan_results.append(last_outer)
            for outer_top_down, pan_block in zip(results[1:], self.pan_blocks):

                if self.drop_block and self.training:
                    pan_lateral = getattr(self, pan_block[0])(self.drop_block(last_outer))
                else:
                    pan_lateral = getattr(self, pan_block[0])(last_outer)

                last_outer = getattr(self, pan_block[1])(pan_lateral + outer_top_down)
                pan_results.append(last_outer)
            results = pan_results

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        try:
            return tuple(results), x_text, swint_feature_c4
        except NameError as e:
            return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


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