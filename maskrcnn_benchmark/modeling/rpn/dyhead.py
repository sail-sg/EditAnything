import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_atss_postprocessor
from .loss import make_atss_loss_evaluator
from .anchor_generator import make_anchor_generator_complex

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.layers import Scale, DYReLU, SELayer, ModulatedDeformConv
from maskrcnn_benchmark.layers import NaiveSyncBatchNorm2d, FrozenBatchNorm2d
from maskrcnn_benchmark.modeling.backbone.fbnet import *


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

        return pred_boxes


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 deformable=False,
                 bn_type=None):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                            groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(out_channels)
        elif bn_type == "sbn":
            bn_op = nn.SyncBatchNorm(out_channels)
        elif bn_type == "nsbn":
            bn_op = NaiveSyncBatchNorm2d(out_channels)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(out_channels)
        if bn_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyConv(torch.nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=nn.Conv2d,
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

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]
        return next_x


class DyHead(torch.nn.Module):
    def __init__(self, cfg):
        super(DyHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.DYHEAD.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS
        if cfg.MODEL.DYHEAD.USE_GN:
            bn_type = ['gn', cfg.MODEL.GROUP_NORM.NUM_GROUPS]
        elif cfg.MODEL.DYHEAD.USE_NSYNCBN:
            bn_type = 'nsbn'
        elif cfg.MODEL.DYHEAD.USE_SYNCBN:
            bn_type = 'sbn'
        else:
            bn_type = None

        use_dyrelu = cfg.MODEL.DYHEAD.USE_DYRELU
        use_dyfuse = cfg.MODEL.DYHEAD.USE_DYFUSE
        use_deform = cfg.MODEL.DYHEAD.USE_DFCONV

        if cfg.MODEL.DYHEAD.CONV_FUNC:
            conv_func = lambda i, o, s: eval(cfg.MODEL.DYHEAD.CONV_FUNC)(i, o, s, bn_type=bn_type)
        else:
            conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and in_channels == channels) if i == 0 else use_dyrelu,
                    use_dyfuse=(use_dyfuse and in_channels == channels) if i == 0 else use_dyfuse,
                    use_deform=(use_deform and in_channels == channels) if i == 0 else use_deform,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))
        if cfg.MODEL.DYHEAD.COSINE_SCALE <= 0:
            self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1)
            self.cls_logits_bias = None
        else:
            self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1, bias=False)
            self.cls_logits_bias = nn.Parameter(torch.zeros(num_anchors * num_classes, requires_grad=True))
            self.cosine_scale = nn.Parameter(torch.ones(1) * cfg.MODEL.DYHEAD.COSINE_SCALE)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if hasattr(l, 'bias') and l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.cls_logits_bias is None:
            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        else:
            torch.nn.init.constant_(self.cls_logits_bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def extract_feature(self, x):
        output = []
        for i in range(len(self.dyhead_tower)):
            x = self.dyhead_tower[i](x)
            output.append(x)
        return output

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []

        dyhead_tower = self.dyhead_tower(x)

        for l, feature in enumerate(x):
            if self.cls_logits_bias is None:
                logit = self.cls_logits(dyhead_tower[l])
            else:
                # CosineSimOutputLayers: https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/modeling/roi_heads/fast_rcnn.py#L448-L464
                # normalize the input x along the `channel` dimension
                x_norm = torch.norm(dyhead_tower[l], p=2, dim=1, keepdim=True).expand_as(dyhead_tower[l])
                x_normalized = dyhead_tower[l].div(x_norm + 1e-5)
                # normalize weight
                temp_norm = (
                    torch.norm(self.cls_logits.weight.data, p=2, dim=1, keepdim=True)
                        .expand_as(self.cls_logits.weight.data)
                )
                self.cls_logits.weight.data = self.cls_logits.weight.data.div(
                    temp_norm + 1e-5
                )
                cos_dist = self.cls_logits(x_normalized)
                logit = self.cosine_scale * cos_dist + self.cls_logits_bias.reshape(1, len(self.cls_logits_bias), 1, 1)
            logits.append(logit)

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower[l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower[l]))
        return logits, bbox_reg, centerness


class DyHeadModule(torch.nn.Module):

    def __init__(self, cfg):
        super(DyHeadModule, self).__init__()
        self.cfg = cfg
        self.head = DyHead(cfg)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder)
        self.box_selector_train = make_atss_postprocessor(cfg, box_coder, is_train=True)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder, is_train=False)
        self.anchor_generator = make_anchor_generator_complex(cfg)

    def forward(self, images, features, targets=None):
        box_cls, box_regression, centerness = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors)
        else:
            return self._forward_test(box_cls, box_regression, centerness, anchors)

    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors):
        loss_box_cls, loss_box_reg, loss_centerness, _, _, _, _ = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        if self.cfg.MODEL.RPN_ONLY:
            return None, losses
        else:
            # boxes = self.box_selector_train(box_cls, box_regression, centerness, anchors)
            boxes = self.box_selector_train(box_regression, centerness, anchors, box_cls)
            train_boxes = []
            # for b, a in zip(boxes, anchors):
            #     a = cat_boxlist(a)
            #     b.add_field("visibility", torch.ones(b.bbox.shape[0], dtype=torch.bool, device=b.bbox.device))
            #     del b.extra_fields['scores']
            #     del b.extra_fields['labels']
            #     train_boxes.append(cat_boxlist([b, a]))
            for b, t in zip(boxes, targets):
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                train_boxes.append(cat_boxlist([b, tb]))
            return train_boxes, losses

    def _forward_test(self, box_cls, box_regression, centerness, anchors):
        boxes = self.box_selector_test(box_regression, centerness, anchors, box_cls)
        return boxes, {}
