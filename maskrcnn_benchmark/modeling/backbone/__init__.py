from collections import OrderedDict

from torch import nn

from .. import registry
from maskrcnn_benchmark.modeling import conv_with_kaiming_uniform
from glip.maskrcnn_benchmark.layers import DropBlock2D, DyHead
from . import fpn as fpn_module
from . import bifpn
from . import resnet
from . import efficientnet
from . import efficientdet
from . import swint
from . import swint_v2
from . import swint_vl
from . import swint_v2_vl


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.BACKBONES.register("R-50-RETINANET")
@registry.BACKBONES.register("R-101-RETINANET")
def build_resnet_c5_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.BACKBONES.register("SWINT-FPN-RETINANET")
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


@registry.BACKBONES.register("SWINT-FPN")
def build_swint_fpn_backbone(cfg):
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
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stages[-4],
            in_channels_stages[-3],
            in_channels_stages[-2],
            in_channels_stages[-1],
            ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
        drop_block=DropBlock2D(cfg.MODEL.FPN.DROP_PROB, cfg.MODEL.FPN.DROP_SIZE) if cfg.MODEL.FPN.DROP_BLOCK else None,
        use_spp=cfg.MODEL.FPN.USE_SPP,
        use_pan=cfg.MODEL.FPN.USE_PAN
    )
    if cfg.MODEL.FPN.USE_DYHEAD:
        dyhead = DyHead(cfg, out_channels)
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("dyhead", dyhead)]))
    else:
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


@registry.BACKBONES.register("CVT-FPN-RETINANET")
def build_retinanet_cvt_fpn_backbone(cfg):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    body = cvt.build_cvt_backbone(cfg)
    in_channels_stages = cfg.MODEL.SPEC.DIM_EMBED
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
        use_pan=cfg.MODEL.FPN.USE_PAN
    )
    if cfg.MODEL.FPN.USE_DYHEAD:
        dyhead = DyHead(cfg, out_channels)
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn), ("dyhead", dyhead)]))
    else:
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


@registry.BACKBONES.register("EFFICIENT7-FPN-RETINANET")
@registry.BACKBONES.register("EFFICIENT7-FPN-FCOS")
@registry.BACKBONES.register("EFFICIENT5-FPN-RETINANET")
@registry.BACKBONES.register("EFFICIENT5-FPN-FCOS")
@registry.BACKBONES.register("EFFICIENT3-FPN-RETINANET")
@registry.BACKBONES.register("EFFICIENT3-FPN-FCOS")
def build_eff_fpn_p6p7_backbone(cfg):
    version = cfg.MODEL.BACKBONE.CONV_BODY.split('-')[0]
    version = version.replace('EFFICIENT', 'b')
    body = efficientnet.get_efficientnet(cfg, version)
    in_channels_stage = body.out_channels
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    in_channels_stage[0] = 0
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_stage,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
        drop_block=DropBlock2D(cfg.MODEL.FPN.DROP_PROB, cfg.MODEL.FPN.DROP_SIZE) if cfg.MODEL.FPN.DROP_BLOCK else None,
        use_spp=cfg.MODEL.FPN.USE_SPP,
        use_pan=cfg.MODEL.FPN.USE_PAN
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


@registry.BACKBONES.register("EFFICIENT7-BIFPN-RETINANET")
@registry.BACKBONES.register("EFFICIENT7-BIFPN-FCOS")
@registry.BACKBONES.register("EFFICIENT5-BIFPN-RETINANET")
@registry.BACKBONES.register("EFFICIENT5-BIFPN-FCOS")
@registry.BACKBONES.register("EFFICIENT3-BIFPN-RETINANET")
@registry.BACKBONES.register("EFFICIENT3-BIFPN-FCOS")
def build_eff_fpn_p6p7_backbone(cfg):
    version = cfg.MODEL.BACKBONE.CONV_BODY.split('-')[0]
    version = version.replace('EFFICIENT', 'b')
    body = efficientnet.get_efficientnet(cfg, version)
    in_channels_stage = body.out_channels
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    bifpns = nn.ModuleList()
    for i in range(cfg.MODEL.BIFPN.NUM_REPEATS):
        first_time = (i==0)
        fpn = bifpn.BiFPN(
            in_channels_list=in_channels_stage[1:],
            out_channels=out_channels,
            first_time=first_time,
            attention=cfg.MODEL.BIFPN.USE_ATTENTION
        )
        bifpns.append(fpn)
    model = nn.Sequential(OrderedDict([("body", body), ("bifpn", bifpns)]))
    return model


@registry.BACKBONES.register("EFFICIENT-DET")
def build_efficientdet_backbone(cfg):
    efficientdet.g_simple_padding = True
    compound = cfg.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND
    start_from = cfg.MODEL.BACKBONE.EFFICIENT_DET_START_FROM
    model = efficientdet.EffNetFPN(
        compound_coef=compound,
        start_from=start_from,
    )
    if cfg.MODEL.BACKBONE.USE_SYNCBN:
        import torch
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
