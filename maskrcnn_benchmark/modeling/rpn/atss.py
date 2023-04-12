import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_atss_postprocessor
from .loss import make_atss_loss_evaluator

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.layers import Scale, DFConv2d, DYReLU, SELayer
from .anchor_generator import make_anchor_generator_complex


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


class ATSSHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ATSSHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ATSS.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.ATSS.CHANNELS
        use_gn = cfg.MODEL.ATSS.USE_GN
        use_bn = cfg.MODEL.ATSS.USE_BN
        use_dcn_in_tower = cfg.MODEL.ATSS.USE_DFCONV
        use_dyrelu = cfg.MODEL.ATSS.USE_DYRELU
        use_se = cfg.MODEL.ATSS.USE_SE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.ATSS.NUM_CONVS):
            if use_dcn_in_tower and \
                    i == cfg.MODEL.ATSS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels if i==0 else channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            if use_gn:
                cls_tower.append(nn.GroupNorm(32, channels))
            if use_bn:
                cls_tower.append(nn.BatchNorm2d(channels))
            if use_se:
                cls_tower.append(SELayer(channels))
            if use_dyrelu:
                cls_tower.append(DYReLU(channels, channels))
            else:
                cls_tower.append(nn.ReLU())

            bbox_tower.append(
                conv_func(
                    in_channels if i == 0 else channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            if use_gn:
                bbox_tower.append(nn.GroupNorm(32, channels))
            if use_bn:
                bbox_tower.append(nn.BatchNorm2d(channels))
            if use_se:
                bbox_tower.append(SELayer(channels))
            if use_dyrelu:
                bbox_tower.append(DYReLU(channels, channels))
            else:
                bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            channels, num_anchors * 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.ATSS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(box_tower))
        return logits, bbox_reg, centerness


class ATSSModule(torch.nn.Module):

    def __init__(self, cfg):
        super(ATSSModule, self).__init__()
        self.cfg = cfg
        self.head = ATSSHead(cfg)
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
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
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
            boxes = self.box_selector_train(box_cls, box_regression, centerness, anchors)
            train_boxes = []
            for b, a in zip(boxes, anchors):
                a = cat_boxlist(a)
                b.add_field("visibility", torch.ones(b.bbox.shape[0], dtype=torch.bool, device=b.bbox.device))
                del b.extra_fields['scores']
                del b.extra_fields['labels']
                train_boxes.append(cat_boxlist([b, a]))
            return train_boxes, losses

    def _forward_test(self, box_cls, box_regression, centerness, anchors):
        boxes = self.box_selector_test(box_cls, box_regression, centerness, anchors)
        return boxes, {}
