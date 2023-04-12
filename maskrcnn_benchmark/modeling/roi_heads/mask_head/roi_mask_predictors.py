# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d, _NewEmptyTensorOp
from maskrcnn_benchmark.layers import ConvTranspose2d
from ...utils import permute_and_flatten


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        # TODO: a hack for binary mask head
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes = 2
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


class VLMaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(VLMaskRCNNC4Predictor, self).__init__()
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)

        # self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
        log_scale = cfg.MODEL.DYHEAD.LOG_SCALE
        self.out_dim = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                     dim_reduced, bias=True)
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, language_dict_features):
        x = F.relu(self.conv5_mask(x))
        if x.numel() <= 0:
            output_shape = [x.shape[0], self.out_dim] + x.shape[-2:]
            return _NewEmptyTensorOp.apply(x, output_shape)

        embedding = language_dict_features["hidden"]
        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang)

        B, C, H, W = x.shape
        # add bias (language)
        dot_product_proj_queries = self.dot_product_projection_image(x)
        dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)
        A = dot_product_proj_queries.shape[1]
        bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)

        # dot product
        dot_product_logit = (torch.matmul(dot_product_proj_queries,
                                          dot_product_proj_tokens.transpose(-1,
                                                                            -2)) / self.log_scale.exp()) + bias
        # clamp for stability
        dot_product_logit = torch.clamp(dot_product_logit, max=50000)
        dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        dot_product_logit = dot_product_logit.view(B, H, W, self.out_dim).permute(0, 3, 1, 2)
        return dot_product_logit


_ROI_MASK_PREDICTOR = {"MaskRCNNC4Predictor": MaskRCNNC4Predictor,
                       "VLMaskRCNNC4Predictor": VLMaskRCNNC4Predictor}


def make_roi_mask_predictor(cfg):
    func = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg)
