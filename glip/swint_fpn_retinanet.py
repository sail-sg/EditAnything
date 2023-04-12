from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.layers import DropBlock2D, DyHead
from . import fpn as fpn_module
from . import swint


def build_retinanet_swint_fpn_backbone(cfg):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.SWINT.VERSION == "v1":
        body = swint.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2":
        body = swint_v2.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "vl":
        body = swint_vl.build_swint_backbone(cfg)
    elif cfg.MODEL.SWINT.VERSION == "v2_vl":
        body = swint_v2_vl.build_swint_backbone(cfg)

    in_channels_stages = cfg.MODEL.SWINT.OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stages[-3],
            in_channels_stages[-2],
            in_channels_stages[-1],
            ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
        drop_block=DropBlock2D(cfg.MODEL.FPN.DROP_PROB, cfg.MODEL.FPN.DROP_SIZE) if cfg.MODEL.FPN.DROP_BLOCK else None,
        use_spp=cfg.MODEL.FPN.USE_SPP,
        use_pan=cfg.MODEL.FPN.USE_PAN,
        return_swint_feature_before_fusion=cfg.MODEL.FPN.RETURN_SWINT_FEATURE_BEFORE_FUSION
    )
    if cfg.MODEL.FPN.USE_DYHEAD:
        dyhead = DyHead(cfg, out_channels)
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("dyhead", dyhead)]))
    else:
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model
