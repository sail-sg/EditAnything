import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


def _rename_conv_weights_for_deformable_conv_layers(state_dict, cfg):
    import re
    layer_keys = sorted(state_dict.keys())
    for ix, stage_with_dcn in enumerate(cfg.MODEL.RESNETS.STAGE_WITH_DCN, 1):
        if not stage_with_dcn:
            continue
        for old_key in layer_keys:
            pattern = ".*block{}.*conv2.*".format(ix)
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
                # Calculate SD conv weight
                w = state_dict[old_key]
                v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
                w = (w - m) / torch.sqrt(v + 1e-10)

                state_dict[new_key] = w
                del state_dict[old_key]
    return state_dict


def load_big_format(cfg, f):
    model = OrderedDict()
    weights = np.load(f)

    cmap = {'a':1, 'b':2, 'c':3}
    for key, val in weights.items():
        old_key = key.replace('resnet/', '')
        if 'root_block' in old_key:
            new_key = 'root.conv.weight'
        elif '/proj/standardized_conv2d/kernel' in old_key:
            key_pattern = old_key.replace('/proj/standardized_conv2d/kernel', '').replace('resnet/', '')
            bname, uname, cidx = key_pattern.split('/')
            new_key = '{}.downsample.{}.conv{}.weight'.format(bname,uname,cmap[cidx])
        elif '/standardized_conv2d/kernel' in old_key:
            key_pattern = old_key.replace('/standardized_conv2d/kernel', '').replace('resnet/', '')
            bname, uname, cidx = key_pattern.split('/')
            new_key = '{}.{}.conv{}.weight'.format(bname,uname,cmap[cidx])
        elif '/group_norm/gamma' in old_key:
            key_pattern = old_key.replace('/group_norm/gamma', '').replace('resnet/', '')
            bname, uname, cidx = key_pattern.split('/')
            new_key = '{}.{}.gn{}.weight'.format(bname,uname,cmap[cidx])
        elif '/group_norm/beta' in old_key:
            key_pattern = old_key.replace('/group_norm/beta', '').replace('resnet/', '')
            bname, uname, cidx = key_pattern.split('/')
            new_key = '{}.{}.gn{}.bias'.format(bname,uname,cmap[cidx])
        else:
            print('Unknown key {}'.format(old_key))
            continue
        print('Map {} -> {}'.format(key, new_key))
        model[new_key] = tf2th(val)

    model = _rename_conv_weights_for_deformable_conv_layers(model, cfg)

    return dict(model=model)
