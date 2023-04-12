from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.poolers import Pooler

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


class KeypointRCNNFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(KeypointRCNNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        input_features = cfg.MODEL.BACKBONE.OUT_CHANNELS
        layers = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
        next_feature = input_features
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "conv_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x

class KeypointRCNNFeature2XZoomExtractor(nn.Module):
    def __init__(self, cfg):
        super(KeypointRCNNFeature2XZoomExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        input_features = cfg.MODEL.BACKBONE.OUT_CHANNELS
        layers = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
        next_feature = input_features
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "conv_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            if layer_idx==len(layers)//2:
                deconv_kernel = 4
                kps_upsacle = ConvTranspose2d(layer_features, layer_features, deconv_kernel,
                                              stride=2, padding=deconv_kernel//2-1)
                nn.init.kaiming_normal_(kps_upsacle.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(kps_upsacle.bias, 0)
                self.add_module("conv_fcn_upscale", kps_upsacle)
                self.blocks.append("conv_fcn_upscale")

            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x


_ROI_KEYPOINT_FEATURE_EXTRACTORS = {
    "KeypointRCNNFeatureExtractor": KeypointRCNNFeatureExtractor,
    "KeypointRCNNFeature2XZoomExtractor": KeypointRCNNFeature2XZoomExtractor
}


def make_roi_keypoint_feature_extractor(cfg):
    func = _ROI_KEYPOINT_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)