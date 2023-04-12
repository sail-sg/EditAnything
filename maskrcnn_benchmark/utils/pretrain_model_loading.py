import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

def _remove_bn_statics(state_dict):
    layer_keys = sorted(state_dict.keys())
    remove_list = []
    for key in layer_keys:
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            remove_list.append(key)
    for key in remove_list:
        del state_dict[key]
    return state_dict

def _rename_conv_weights_for_deformable_conv_layers(state_dict, cfg):
    import re
    layer_keys = sorted(state_dict.keys())
    for ix, stage_with_dcn in enumerate(cfg.MODEL.RESNETS.STAGE_WITH_DCN, 1):
        if not stage_with_dcn:
            continue
        for old_key in layer_keys:
            pattern = ".*layer{}.*conv2.*".format(ix)
            r = re.match(pattern, old_key)
            if r is None:
                continue
            for param in ["weight", "bias"]:
                if old_key.find(param) is -1:
                    continue
                if 'unit01' in old_key:
                    continue
                new_key = old_key.replace(
                    "conv2.{}".format(param), "conv2.conv.{}".format(param)
                )
                print("pattern: {}, old_key: {}, new_key: {}".format(
                    pattern, old_key, new_key
                ))
                state_dict[new_key] = state_dict[old_key]
                del state_dict[old_key]
    return state_dict


def load_pretrain_format(cfg, f):
    model = torch.load(f)
    model = _remove_bn_statics(model)
    model = _rename_conv_weights_for_deformable_conv_layers(model, cfg)

    return dict(model=model)
