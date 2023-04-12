# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import inspect
import logging

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, pairwise_ioa
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.sampling import subsample_labels

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient

from .vlpart_fast_rcnn import VLMFastRCNNOutputLayers


def build_vlpart_roi_heads(cfg, input_shape):
    return CascadeVLMROIHeads(cfg, input_shape)


class CascadeVLMROIHeads(CascadeROIHeads):

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, height=pooler_resolution, width=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                VLMFastRCNNOutputLayers(
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }

    def forward(self, images, features, proposals, text_embed):
        del images
        assert not self.training, 'only support inference now'
        pred_instances = self._forward_box(
            features, proposals, text_embed=text_embed)
        pred_instances = self.forward_with_given_boxes(features, pred_instances)
        return pred_instances, {}

    def _forward_box(self, features, proposals, text_embed):

        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes)
                if self.training and ann_type in ['box', 'part']:
                    proposals = self._match_and_label_boxes(
                        proposals, k, targets)
            predictions = self._run_stage(features, proposals, k, text_embed)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        assert not self.training, 'only support inference now'
        # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
        scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
        scores = [
            sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
            for scores_per_image in zip(*scores_per_stage)
        ]
        predictor, predictions, proposals = head_outputs[-1]
        boxes = predictor.predict_boxes((predictions[0], predictions[1]), proposals)
        pred_instances, _ = fast_rcnn_inference(
            boxes,
            scores,
            image_sizes,
            predictor.test_score_thresh,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )
        return pred_instances

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

    def _run_stage(self, features, proposals, stage, text_embed):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](box_features, text_embed)
