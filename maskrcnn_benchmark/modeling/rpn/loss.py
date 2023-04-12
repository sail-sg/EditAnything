# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat, concat_box_prediction_layers

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.layers import SigmoidFocalLoss, IOULoss, TokenSigmoidFocalLoss
from maskrcnn_benchmark.utils.comm import get_world_size, reduce_sum
from maskrcnn_benchmark.utils.amp import custom_fwd, custom_bwd
from maskrcnn_benchmark.utils.shallow_contrastive_loss_helper import *

from transformers import AutoTokenizer

INF = 1e8


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds

        if len(target):
            matched_targets = target[matched_idxs.clamp(min=0)]
        else:
            matched_targets = target

        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            if not matched_targets.bbox.shape[0]:
                zeros = torch.zeros_like(labels_per_image)
                regression_targets_per_image = torch.stack((zeros, zeros, zeros, zeros), dim=1)
            else:
                regression_targets_per_image = self.box_coder.encode(matched_targets.bbox, anchors_per_image.bbox)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    @custom_fwd(cast_inputs=torch.float32)
    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level in zip(
                objectness, box_regression
        ):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


class FocalLossComputation(object):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    @custom_fwd(cast_inputs=torch.float32)
    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = \
            concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))

        labels = labels.int()

        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels
        ) / (pos_inds.numel() + N)

        return retinanet_cls_loss, retinanet_regression_loss


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FOCAL.LOSS_GAMMA,
            cfg.MODEL.FOCAL.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.use_gt_center = cfg.MODEL.FCOS.USE_GT_CENTER

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = torch.nn.BCEWithLogitsLoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"

            if self.use_gt_center:
                center = targets_per_im.get_field("cbox")
                bboxes = center.bbox
                area = center.area()
            else:
                bboxes = targets_per_im.bbox
                area = targets_per_im.area()
            labels_per_im = targets_per_im.get_field("labels")

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    @custom_fwd(cast_inputs=torch.float32)
    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / max(pos_inds.numel(), 1.0)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / centerness_targets.sum()
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / max(pos_inds.numel(), 1.0)
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


# class ATSSLossComputation(object):
class ATSSLossComputation(torch.nn.Module):

    def __init__(self, cfg, box_coder):
        super(ATSSLossComputation, self).__init__()
        
        self.cfg = cfg
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.FOCAL.LOSS_GAMMA, cfg.MODEL.FOCAL.LOSS_ALPHA)
        self.centerness_loss_func = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.matcher = Matcher(cfg.MODEL.FOCAL.FG_IOU_THRESHOLD, cfg.MODEL.FOCAL.BG_IOU_THRESHOLD, True)
        self.box_coder = box_coder

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS or self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            self.token_loss_func = TokenSigmoidFocalLoss(cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_ALPHA,
                                                         cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_GAMMA)

        self.lang = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE

        # self.tokenizer = AutoTokenizer.from_pretrained(self.lang)
        if self.cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            # self.tokenizer = build_tokenizer(self.cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                            from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.lang)

        # if use shallow contrastive loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS \
                or self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS:
                assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS == False
                channels = cfg.MODEL.DYHEAD.CHANNELS
                num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
                shallow_input_dim = channels * num_anchors
            elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
                assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS == False
                shallow_input_dim = cfg.MODEL.SWINT.OUT_CHANNELS[-2]

            shallow_log_scale = self.cfg.MODEL.DYHEAD.SHALLOW_LOG_SCALE
            shallow_contrastive_hdim = cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_HIDDEN_DIM
            # self.shallow_contrastive_projection_image = nn.Conv2d(channels, num_anchors * shallow_contrastive_hdim,
            #                                                       kernel_size=1)
            self.shallow_contrastive_projection_image = nn.Linear(shallow_input_dim, shallow_contrastive_hdim,
                                                                  bias=True)
            self.shallow_contrastive_projection_text = nn.Linear(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                                 shallow_contrastive_hdim, bias=True)
            self.shallow_log_scale = nn.Parameter(torch.Tensor([shallow_log_scale]), requires_grad=True)

        # (initialization) if use shallow contrastive loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS:
            for modules in [self.shallow_contrastive_projection_image, self.shallow_contrastive_projection_text]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)
                    if isinstance(l, nn.Linear):
                        torch.nn.init.xavier_uniform_(l.weight)
                        l.bias.data.fill_(0)

    def NllSoftMaxLoss(self, logits, target):
        loss_ce = -target * logits.log_softmax(
            -1)  # basically, only the those positives with positive target_sim will have losses
        return loss_ce

    def ContrastiveAlignLoss(self, logits, positive_map):
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return tot_loss

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def prepare_targets(self, targets, anchors, tokenized=None, positive_map=None, proj_tokens=None):
        cls_labels = []
        reg_targets = []
        token_labels = []
        map_labels = []

        gold_box_od_labels = []
        od_label_of_tokens_labels = []
        positive_indices = []

        offset = 0

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            # bboxes_per_im = targets_per_im.get_field("boxes")
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            num_gt = len(bboxes_per_im)

            if positive_map is not None:
                token_per_im = positive_map[offset:offset + num_gt, :]
                offset += num_gt

            # Recheck if the label matches with the positive map
            # print(labels_per_im)
            # print(token_per_im.nonzero())

            # shallow contrastive
            if "original_od_label" in targets_per_im.fields():
                gold_box_od_label = targets_per_im.get_field("original_od_label")
            if "positive_map_for_od_labels" in targets_per_im.fields():
                od_label_of_token_per_im = targets_per_im.get_field("positive_map_for_od_labels")

            # print(gold_box_od_label)
            # print(od_label_of_token_per_im)

            if positive_map is not None and proj_tokens is not None:
                if "tokens_positive" in targets_per_im.fields():
                    cur_tokens = targets_per_im.get_field("tokens_positive")
                else:
                    cur_tokens = targets_per_im.get_field("tokens")
                map = torch.zeros((len(cur_tokens), proj_tokens.shape[1]), dtype=torch.bool)
                for j, tok_list in enumerate(cur_tokens):
                    for (beg, end) in tok_list:
                        beg_pos = tokenized.char_to_token(im_i, beg)
                        end_pos = tokenized.char_to_token(im_i, end - 1)
                        if beg_pos is None:
                            try:
                                beg_pos = tokenized.char_to_token(im_i, beg + 1)
                                if beg_pos is None:
                                    beg_pos = tokenized.char_to_token(im_i, beg + 2)
                            except:
                                beg_pos = None
                        if end_pos is None:
                            try:
                                end_pos = tokenized.char_to_token(im_i, end - 2)
                                if end_pos is None:
                                    end_pos = tokenized.char_to_token(im_i, end - 3)
                            except:
                                end_pos = None
                        if beg_pos is None or end_pos is None:
                            continue

                        assert beg_pos is not None and end_pos is not None
                        map[j, beg_pos: end_pos + 1].fill_(True)

            anchors_per_im = cat_boxlist(anchors[im_i])

            num_anchors_per_loc = len(self.cfg.MODEL.RPN.ASPECT_RATIOS) * self.cfg.MODEL.RPN.SCALES_PER_OCTAVE
            num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
            ious = boxlist_iou(anchors_per_im, targets_per_im)

            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # Selecting candidates based on the center distance between anchor box and object
            candidate_idxs = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors[im_i]):
                end_idx = star_idx + num_anchors_per_level[level]
                distances_per_level = distances[star_idx:end_idx, :]
                topk = min(self.cfg.MODEL.ATSS.TOPK * num_anchors_per_loc, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)
            l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
            t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            # get positive anchors index from ATSS
            positive_index = [i[0].item() for i in torch.nonzero(anchors_to_gt_indexs)]
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0

            if positive_map is not None:
                token_labels_per_im = token_per_im[anchors_to_gt_indexs]
                unmatched_labels = torch.zeros(token_labels_per_im.shape[1], device=token_labels_per_im.device)
                # TODO: temporarially disable the [NoObj] token logic, and only restrict to binary loss
                unmatched_labels[-1] = 1  # token: none object - > 256
                token_labels_per_im[anchors_to_gt_values == -INF] = unmatched_labels
                # move from cpu to gpu
                token_labels_per_im = token_labels_per_im.to(cls_labels_per_im.device)

                # print(token_labels_per_im[anchors_to_gt_values == -INF].shape)
                # print(cls_labels_per_im[anchors_to_gt_values != -INF][0])
                # print(token_labels_per_im[anchors_to_gt_values != -INF][0].nonzero())

            if positive_map is not None and proj_tokens is not None:
                map_labels_per_im = map[anchors_to_gt_indexs]
                unmatched_labels = torch.zeros(map_labels_per_im.shape[1], dtype=torch.bool,
                                               device=map_labels_per_im.device)  # map: none False
                map_labels_per_im[anchors_to_gt_values == -INF] = unmatched_labels
                # move from cpu to gpu
                map_labels_per_im = map_labels_per_im.to(cls_labels_per_im.device)

                # print(map_labels_per_im[anchors_to_gt_values == -INF].shape)
                # print(map_labels_per_im[anchors_to_gt_values != -INF][0])

            if positive_map is not None and proj_tokens is not None:
                gold_box_od_label_per_im = gold_box_od_label[anchors_to_gt_indexs]
                gold_box_od_label_per_im[anchors_to_gt_values == -INF] = -100
                # move from cpu to gpu
                gold_box_od_label_per_im = gold_box_od_label_per_im.to(cls_labels_per_im.device)

                # print(gold_box_od_label_per_im[anchors_to_gt_values != -INF])

            matched_gts = bboxes_per_im[anchors_to_gt_indexs]

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

            if positive_map is not None:
                token_labels.append(token_labels_per_im)

            if positive_map is not None and proj_tokens is not None:
                map_labels.append(map_labels_per_im)
                gold_box_od_labels.append(gold_box_od_label_per_im)
                od_label_of_tokens_labels.append(od_label_of_token_per_im)
                positive_indices.append(positive_index)

        # print([len(x) for x in positive_indices])

        return cls_labels, reg_targets, token_labels, map_labels, gold_box_od_labels, od_label_of_tokens_labels, positive_indices

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    @custom_fwd(cast_inputs=torch.float32)
    def __call__(self, box_cls, box_regression, centerness, targets, anchors,
                 captions=None,
                 positive_map=None,
                 token_logits=None,
                 proj_tokens=None,
                 contrastive_logits=None,
                 dot_product_logits=None,
                 text_masks=None,
                 shallow_img_emb_feats=None
                 ):

        tokenized = None
        if captions is not None:
            # tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt")
            if self.cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
                tokenized = self.tokenizer.batch_encode_plus(captions,
                                                             max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                             padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                             return_tensors='pt',
                                                             truncation=True)
            else:
                tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt")

        labels, reg_targets, token_labels, map_labels, gold_box_od_labels, od_label_of_tokens_labels, positive_indices = self.prepare_targets(targets, anchors,
                                                                             tokenized,
                                                                             positive_map,
                                                                             proj_tokens
                                                                             )

        N = len(labels)

        box_regression_flatten, box_cls_flatten, token_logits_stacked = concat_box_prediction_layers(
            box_regression,
            box_cls,
            token_logits,
        )

        # contrastive logits
        if positive_map is not None and contrastive_logits is not None:
            contrastive_logits = torch.cat(contrastive_logits, dim=1)

        # dot product soft token logits
        if dot_product_logits is not None:
            dot_product_logits = torch.cat(dot_product_logits, dim=1)

        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in centerness]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        if positive_map is not None:
            token_labels_stacked = torch.stack(token_labels, dim=0)

        if positive_map is not None and proj_tokens is not None:
            positive_map_box_to_self_text = None
            shallow_positive_map = None
            bs = proj_tokens.shape[0]
            device = proj_tokens.device

            # NOTE: 0. setup env
            if dist.is_dist_avail_and_initialized():
                world_size = dist.get_world_size()
                rank = torch.distributed.get_rank()
            else:
                world_size = 1
                rank = 0

            if contrastive_logits is not None:
                positive_map_box_to_self_text = torch.stack(map_labels, dim=0)

            if shallow_img_emb_feats is not None:
                '''
                Ultimate:
                    N*B*(max_anchor_num) x N*B*T
                Final Goal:
                    F = B x (max_anchor_num) x N*B*T
                        X: B x (max_anchor_num) od_labels : [0, 20, 30, ..]
                        Y: N*B*T: which denotes the od_label of every token
                    F[i,j] = A[i] == B[j]
                '''
                with torch.no_grad():
                    # NOTE: 1. get X (predicted_box_od_label), which the detection label of every predicted boxes
                    # predicted_box_od_label: B x A

                    # check memory limitation: prevent # of positive >= # of max_positive
                    new_positive_indices = []
                    # print([len(positive_index) for positive_index in positive_indices])
                    for positive_index in positive_indices:
                        if len(positive_index) >= self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_MAX_POSITIVE_ANCHORS:
                            import random
                            positive_index = sorted(random.sample(positive_index,
                                                           self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_MAX_POSITIVE_ANCHORS))
                        new_positive_indices.append(positive_index)
                    # print([len(positive_index) for positive_index in positive_indices])

                    max_len = max([len(positive_index) for positive_index in new_positive_indices])
                    max_anchor_num = max_len

                    if world_size > 1:
                        num_anchors = torch.tensor(max_len, device=positive_map.device)
                        num_anchors_full = [torch.zeros_like(num_anchors) for _ in range(world_size)]
                        torch.distributed.all_gather(num_anchors_full, num_anchors)
                        max_anchor_num = max([anchor.item() for anchor in num_anchors_full])

                    new_negative_pad_indices = []
                    # if not PAD_ZEROS, select random negative paddings
                    if not self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_ZERO_PADS:
                        for (positive_index, old_positive_index) in zip(new_positive_indices, positive_indices):
                            negative_index = [i for i in range(len(cat_boxlist(anchors[0]))) if i not in old_positive_index]
                            import random
                            negative_pad_index = sorted(random.sample(negative_index,
                                                               max_anchor_num - len(positive_index)))
                            new_negative_pad_indices.append(negative_pad_index)

                    predicted_box_od_label = []
                    for i in range(bs):
                        predicted_box_od_label.append(
                            pad_tensor_given_dim_length(gold_box_od_labels[i][new_positive_indices[i]],
                                                        dim=0,
                                                        length=max_anchor_num,
                                                        padding_value=-100,
                                                        batch_first=False
                                                        ))
                    predicted_box_od_label = torch.stack(predicted_box_od_label, dim=0)

                    # if padding, need to create image masks to filter out the paddings
                    image_masks = None
                    if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_ZERO_PADS:
                        image_masks = torch.zeros((bs, max_anchor_num), dtype=torch.long).to(text_masks.device)
                        for i in range(bs):
                            image_masks[i, :len(new_positive_indices[i])] = 1

                    # NOTE: 2. Get Y (od_label_of_tokens)
                    # od_label_of_tokens: N x B x T
                    od_label_of_tokens = torch.stack(od_label_of_tokens_labels, dim=0).long()
                    od_label_of_tokens = gather_tensors(od_label_of_tokens)

                    # NOTE: 3. get F
                    # F: B*A x N*B*T
                    mapping_predicted_box_to_all_text = predicted_box_od_label.view(-1).unsqueeze(
                        1) == od_label_of_tokens.view(-1).unsqueeze(0)

                    # NOTE: 4. we still need to calculate the mapping between predicted box to its corresponding text's mapping
                    # positive_map_box_to_self_text: B x A x T, leave this for vanilla contrastive alignment loss
                    positive_map_box_to_self_text = []
                    for i in range(bs):
                        positive_map_box_to_self_text.append(
                            pad_tensor_given_dim_length(map_labels[i][new_positive_indices[i]],
                                                        dim=0,
                                                        length=max_anchor_num,
                                                        padding_value=False,
                                                        batch_first=False
                                                        ))
                    positive_map_box_to_self_text = torch.stack(positive_map_box_to_self_text, dim=0)

                    # change the corresponding place in our batch
                    for i in range(bs):
                        mapping_predicted_box_to_all_text[i * max_anchor_num: (i + 1) * max_anchor_num,
                        (rank * bs + i) * 256: (rank * bs + i + 1) * 256] = positive_map_box_to_self_text[i]

                    # NOTE: 5. communicate and get positive map
                    # mapping_predicted_box_to_all_text: N*B*A x N*B*T
                    mapping_predicted_box_to_all_text = gather_tensors(mapping_predicted_box_to_all_text).view(-1,
                                                                                                               mapping_predicted_box_to_all_text.size(
                                                                                                                   -1))
                    shallow_positive_map = mapping_predicted_box_to_all_text  # This is the true positive map
                    shallow_positive_map = shallow_positive_map.unsqueeze(0)

                    # Get text attention masks
                    text_attention_mask = torch.zeros((bs, 256), dtype=torch.long)  # B x 256
                    for i in range(bs):
                        text_attention_mask[i, :len(text_masks[i])] = text_masks[i]
                    text_attention_mask = gather_tensors(
                        text_attention_mask.bool().to(device))  # N x B x 256

                    # if PAD_ZEROS, get image masks
                    if image_masks is not None:
                        image_attention_mask = torch.zeros((bs, max_anchor_num), dtype=torch.long)  # B x max_anchor
                        for i in range(bs):
                            image_attention_mask[i, :len(image_masks[i])] = image_masks[i]
                        image_attention_mask = gather_tensors(
                            image_attention_mask.bool().to(device))  # N x B x max_anchor

                # NOTE: 6. calculate shallow contrastive logits
                shallow_proj_tokens = F.normalize(self.shallow_contrastive_projection_text(proj_tokens), p=2, dim=-1)

                shallow_normalized_img_embs = []
                if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
                    # choice 1：use features from SWINT backbone layer (c4) before vl fusion
                    from maskrcnn_benchmark.layers.roi_align import ROIAlignV2
                    pooler = ROIAlignV2((1, 1), 1./16, 0)
                    # get positive features
                    for i in range(bs):
                        rois = convert_to_roi_format(cat_boxlist(anchors[i])[new_positive_indices[i]])
                        roi_feature = pooler(shallow_img_emb_feats[i].unsqueeze(0), rois)
                        roi_feature = roi_feature.squeeze(-1).squeeze(-1)
                        shallow_contrastive_proj_queries = self.shallow_contrastive_projection_image(roi_feature)
                        shallow_normalized_img_emb = F.normalize(shallow_contrastive_proj_queries, p=2, dim=-1)
                        if image_masks is not None:
                            # pad zeros
                            shallow_normalized_img_embs.append(
                                pad_tensor_given_dim_length(shallow_normalized_img_emb,
                                                            dim=0,
                                                            length=max_anchor_num,
                                                            padding_value=0.0,
                                                            batch_first=False
                                                            ))
                        else:
                            # pad negatives
                            negative_rois = convert_to_roi_format(cat_boxlist(anchors[i])[new_negative_pad_indices[i]])
                            negative_roi_feature = pooler(shallow_img_emb_feats[i].unsqueeze(0), negative_rois)
                            negative_roi_feature = negative_roi_feature.squeeze(-1).squeeze(-1)
                            negative_shallow_contrastive_proj_queries = self.shallow_contrastive_projection_image(negative_roi_feature)
                            negative_shallow_normalized_img_emb = F.normalize(negative_shallow_contrastive_proj_queries,
                                                                              p=2, dim=-1)
                            shallow_normalized_img_embs.append(
                                pad_random_negative_tensor_given_length(shallow_normalized_img_emb,
                                                                        negative_shallow_normalized_img_emb,
                                                                        length=max_anchor_num
                                                                        )
                            )
                elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS:
                    # choice 2：use features after FPN
                    shallow_img_embs = torch.cat(shallow_img_emb_feats, dim=1)
                    # get positive features
                    for i in range(bs):
                        shallow_contrastive_proj_queries = self.shallow_contrastive_projection_image(shallow_img_embs[i, new_positive_indices[i], :])
                        shallow_normalized_img_emb = F.normalize(shallow_contrastive_proj_queries, p=2, dim=-1)
                        if image_masks is not None:
                            # pad zeros
                            shallow_normalized_img_embs.append(
                                pad_tensor_given_dim_length(shallow_normalized_img_emb,
                                                            dim=0,
                                                            length=max_anchor_num,
                                                            padding_value=0.0,
                                                            batch_first=False
                                                            ))
                        else:
                            # pad negatives
                            negative_shallow_contrastive_proj_queries = self.shallow_contrastive_projection_image(shallow_img_embs[i, new_negative_pad_indices[i], :])
                            negative_shallow_normalized_img_emb = F.normalize(negative_shallow_contrastive_proj_queries,
                                                                              p=2, dim=-1)
                            shallow_normalized_img_embs.append(
                                pad_random_negative_tensor_given_length(shallow_normalized_img_emb,
                                                                        negative_shallow_normalized_img_emb,
                                                                        length=max_anchor_num
                                                                        )
                            )

                shallow_normalized_img_embs = torch.stack(shallow_normalized_img_embs, dim=0)
                shallow_normalized_text_emb = shallow_proj_tokens
                shallow_normalized_text_emb = pad_tensor_given_dim_length(shallow_normalized_text_emb,
                                                                          dim=1,
                                                                          length=256,
                                                                          padding_value=0.0)

                gathered_shallow_normalized_img_emb = gather_tensors(shallow_normalized_img_embs)
                gathered_shallow_normalized_text_emb = gather_tensors(shallow_normalized_text_emb)
                gathered_shallow_normalized_img_emb = gathered_shallow_normalized_img_emb.view(-1,
                                                                                               gathered_shallow_normalized_img_emb.size(
                                                                                                   -1))
                gathered_shallow_normalized_text_emb = gathered_shallow_normalized_text_emb.view(-1,
                                                                                                 gathered_shallow_normalized_text_emb.size(
                                                                                                     -1))
                shallow_contrastive_logits = (
                        torch.matmul(gathered_shallow_normalized_img_emb,
                                     gathered_shallow_normalized_text_emb.transpose(-1,
                                                                                    -2)) / self.shallow_log_scale.exp())
                shallow_contrastive_logits = shallow_contrastive_logits.unsqueeze(0)

                # apply text mask
                text_attention_mask = text_attention_mask.view(-1).unsqueeze(0).unsqueeze(0)
                text_attention_mask = text_attention_mask.repeat(1, shallow_contrastive_logits.size(1),
                                                                 1)  # copy along the image feature dimension
                shallow_contrastive_logits = shallow_contrastive_logits.masked_fill(~text_attention_mask, -1000000)

                # if PAD ZEROS, apply image mask
                if image_masks is not None:
                    image_attention_mask = image_attention_mask.view(-1).unsqueeze(0).unsqueeze(-1)
                    image_attention_mask = image_attention_mask.repeat(1, 1, shallow_contrastive_logits.size(
                        2))  # copy along the text feature dimension
                    shallow_contrastive_logits = shallow_contrastive_logits.masked_fill(~image_attention_mask, -1000000)

                # Note: 7. calculate image and text logits and maps
                shallow_image_logits = shallow_contrastive_logits[:,
                                       (rank * bs) * max_anchor_num: (rank * bs + bs) * max_anchor_num, :]
                shallow_image_positive_map = normalized_positive_map(
                    shallow_positive_map[:, (rank * bs) * max_anchor_num: (rank * bs + bs) * max_anchor_num, :])

                shallow_text_logits = shallow_contrastive_logits[:, :,
                                      (rank * bs) * 256: (rank * bs + bs) * 256].transpose(1,
                                                                                           2)
                shallow_text_positive_map = normalized_positive_map(
                    shallow_positive_map[:, :, (rank * bs) * 256: (rank * bs + bs) * 256].transpose(1, 2))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()) / num_pos_avg_per_gpu

        token_logits_loss = None
        contrastive_align_loss = None
        dot_product_token_loss = None
        shallow_contrastive_loss = None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            token_logits_loss = self.token_loss_func(token_logits_stacked,
                                                     token_labels_stacked, text_masks=text_masks,
                                                     version="binary") / num_pos_avg_per_gpu

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            contrastive_align_loss = self.ContrastiveAlignLoss(contrastive_logits, positive_map_box_to_self_text) / num_pos_avg_per_gpu

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            dot_product_token_loss = self.token_loss_func(dot_product_logits,
                                                          token_labels_stacked, text_masks=text_masks,
                                                          version="binary") / num_pos_avg_per_gpu

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS or \
                self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            box_to_token_loss = self.NllSoftMaxLoss(shallow_image_logits, shallow_image_positive_map).sum()
            token_to_box_loss = self.NllSoftMaxLoss(shallow_text_logits, shallow_text_positive_map).sum()
            tot_loss = (box_to_token_loss + token_to_box_loss) / 2
            shallow_contrastive_loss = tot_loss / num_pos_avg_per_gpu

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)

            sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss * self.cfg.MODEL.ATSS.REG_LOSS_WEIGHT, centerness_loss, \
               token_logits_loss, \
               contrastive_align_loss, \
               dot_product_token_loss, \
               shallow_contrastive_loss


def generate_anchor_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image


def make_focal_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.FOCAL.FG_IOU_THRESHOLD,
        cfg.MODEL.FOCAL.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    sigmoid_focal_loss = SigmoidFocalLoss(
        cfg.MODEL.FOCAL.LOSS_GAMMA,
        cfg.MODEL.FOCAL.LOSS_ALPHA
    )

    loss_evaluator = FocalLossComputation(
        matcher,
        box_coder,
        generate_anchor_labels,
        sigmoid_focal_loss,
        bbox_reg_beta=cfg.MODEL.FOCAL.BBOX_REG_BETA,
        regress_norm=cfg.MODEL.FOCAL.BBOX_REG_WEIGHT,
    )
    return loss_evaluator


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator


def make_atss_loss_evaluator(cfg, box_coder):
    loss_evaluator = ATSSLossComputation(cfg, box_coder)
    return loss_evaluator
