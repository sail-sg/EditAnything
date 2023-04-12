import logging
import numpy as np
import itertools
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like, batched_nms
from detectron2.structures import ImageList, Boxes, Instances, BitMasks, ROIMasks

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.config import get_cfg


import clip
from vlpart.text_encoder import build_text_encoder
from vlpart.swintransformer import build_swinbase_fpn_backbone
from vlpart.vlpart_roi_heads import build_vlpart_roi_heads


def build_vlpart(checkpoint=None):
    cfg = get_cfg()
    cfg.merge_from_list(['MODEL.RPN.IN_FEATURES', ["p2", "p3", "p4", "p5", "p6"],
                         'MODEL.ROI_HEADS.IN_FEATURES', ["p2", "p3", "p4", "p5"],
                         'MODEL.ROI_BOX_CASCADE_HEAD.IOUS', [0.5, 0.6, 0.7],
                         'MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG', True,
                         'MODEL.ROI_BOX_HEAD.NAME', "FastRCNNConvFCHead",
                         'MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION', 7,
                         'MODEL.ROI_BOX_HEAD.NUM_FC', 2,
                         'MODEL.ANCHOR_GENERATOR.SIZES', [[32], [64], [128], [256], [512]],
                         'MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS', [[0.5, 1.0, 2.0]],
    ])
    backbone = build_swinbase_fpn_backbone()
    vlpart = VLPart(
        backbone=backbone,
        proposal_generator=build_proposal_generator(cfg, backbone.output_shape()),
        roi_heads=build_vlpart_roi_heads(cfg, backbone.output_shape()),
    )
    vlpart.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        vlpart.load_state_dict(state_dict['model'], strict=False)

    return vlpart


class VLPart(nn.Module):
    def __init__(
        self,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.text_encoder = build_text_encoder(pretrain=True, visual_type='RN50')

        self.register_buffer("pixel_mean",
                             torch.tensor([123.675, 116.280, 103.530]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std",
                             torch.tensor([58.395, 57.120, 57.375]).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def get_text_embeddings(self, vocabulary, prefix_prompt='a '):
        vocabulary = vocabulary.split('.')
        texts = [prefix_prompt + x.lower().replace(':', ' ') for x in vocabulary]
        texts_aug = texts + ['background']
        emb = self.text_encoder(texts_aug).permute(1, 0)
        emb = F.normalize(emb, p=2, dim=0)
        return emb

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        do_postprocess: bool = True,
        text_prompt: str = 'dog',
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features)
        text_embed = self.get_text_embeddings(text_prompt)
        results, _ = self.roi_heads(images, features, proposals, text_embed)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            max_shape = images.tensor.shape[2:]
            return VLPart._postprocess(results, batched_inputs, images.image_sizes, max_shape)
        else:
            return results


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in original_images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes, max_shape):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = custom_detector_postprocess(results_per_image, height, width, max_shape)
            processed_results.append({"instances": r})
        return processed_results


def custom_detector_postprocess(
        results: Instances, output_height: int, output_width: int,
        max_shape, mask_threshold: float = 0.5
):
    """
    detector_postprocess with support on global_masks
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )

    resized_h, resized_w = results.image_size
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    return results
