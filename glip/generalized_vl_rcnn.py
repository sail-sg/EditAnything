# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from transformers import AutoTokenizer

import random
import timeit
from copy import deepcopy

from .swint_fpn_retinanet import build_retinanet_swint_fpn_backbone
from .bert_model import build_bert_backbone
from .vldyhead import build_rpn_vldyhead


def build_generalized_vlrcnn_model(cfg):
    return GeneralizedVLRCNN(cfg)


class GeneralizedVLRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        self.cfg = cfg

        # visual encoder
        self.backbone = build_retinanet_swint_fpn_backbone(cfg)

        # language encoder
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]
        self.language_backbone = build_bert_backbone(cfg)

        # fuse
        self.rpn = build_rpn_vldyhead(cfg)

    def forward(self,
                images,
                targets=None,
                captions=None,
                positive_map=None,
                greenlight_map=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device

        # language embedding
        language_dict_features = {}
        if captions is not None:
            # print(captions[0])
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                         max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                         padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                         return_special_tokens_mask=True,
                                                         return_tensors='pt',
                                                         truncation=True).to(device)

            input_ids = tokenized.input_ids
            mlm_labels = None

            tokenizer_input = {"input_ids": input_ids,
                               "attention_mask": tokenized.attention_mask}

            language_dict_features = self.language_backbone(tokenizer_input)

            language_dict_features["mlm_labels"] = mlm_labels

        # visual embedding
        swint_feature_c4 = None
        visual_features = self.backbone(images.tensors)

        # fuse and predict
        proposals, proposal_losses, fused_visual_features = \
            self.rpn(images, visual_features, targets,
                     language_dict_features, positive_map,
                     captions, swint_feature_c4)

        result = proposals

        return result
