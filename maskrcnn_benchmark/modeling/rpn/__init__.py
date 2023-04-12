# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from .rpn import build_rpn
from .rpn import RPNModule
from .retina import RetinaNetModule
from .fcos import FCOSModule
from .atss import ATSSModule
from .dyhead import DyHeadModule
from .vldyhead import VLDyHeadModule

_RPN_META_ARCHITECTURES = {"RPN": RPNModule,
                           "RETINA": RetinaNetModule,
                           "FCOS": FCOSModule,
                           "ATSS": ATSSModule,
                           "DYHEAD": DyHeadModule,
                           "VLDYHEAD": VLDyHeadModule
                           }


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    rpn_arch = _RPN_META_ARCHITECTURES[cfg.MODEL.RPN_ARCHITECTURE]
    return rpn_arch(cfg)
