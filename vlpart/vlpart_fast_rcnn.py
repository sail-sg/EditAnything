# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import copy
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss, smooth_l1_loss
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances, BitMasks, pairwise_iou, pairwise_ioa
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class TexeEmbedClassifier(nn.Module):
    def __init__(
        self,
        input_shape: ShapeSpec,
        zs_weight_dim: int = 1024,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.linear = nn.Linear(input_size, zs_weight_dim)

    def forward(self, x, text_embed):

        x = self.linear(x)
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, text_embed)
        return x


class VLMFastRCNNOutputLayers(nn.Module):
    def __init__(
        self,
        input_shape: ShapeSpec,
        box2box_transform,
        use_sigmoid_ce: bool = True,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)

        self.box2box_transform = box2box_transform
        self.use_sigmoid_ce = use_sigmoid_ce
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        input_size = input_shape.channels * \
                     (input_shape.width or 1) * (input_shape.height or 1)

        # bbox_pred
        self.bbox_pred = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, 4)
        )
        # cls_score
        self.cls_score = TexeEmbedClassifier(input_shape)

    def forward(self, x, text_embed):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        cls_scores = self.cls_score(x, text_embed)
        proposal_deltas = self.bbox_pred(x)

        return cls_scores, proposal_deltas

    def predict_boxes(self, predictions, proposals):
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)

        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        cls_scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)

        final_scores = []
        for cls_score in cls_scores:
            final_score = cls_score.sigmoid() if self.use_sigmoid_ce else F.softmax(cls_score, dim=-1)
            final_scores.append(final_score)
        return final_scores

