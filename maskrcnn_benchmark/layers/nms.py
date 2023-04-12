# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark import _C

try:
    import torchvision
    from torchvision.ops import nms
except:
    nms = _C.nms

ml_nms = _C.ml_nms
soft_nms = _C.soft_nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
