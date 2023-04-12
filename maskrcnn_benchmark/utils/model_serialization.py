# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict, defaultdict
import logging
import math
import torch

from maskrcnn_benchmark.utils.imports import import_file

def resize_2d(posemb, shape_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = shape_new[0]
    gs_old = int(math.sqrt(len(posemb)))  # 2 * w - 1
    gs_new = int(math.sqrt(ntok_new))  # 2 * w - 1
    posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(gs_new * gs_new, -1)
    return posemb_grid

def align_and_update_state_dicts(model_state_dict, loaded_state_dict, reshape_keys=['pos_bias_table'], use_weightmap=False):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    matched_keys = []
    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if model_state_dict[key].shape != loaded_state_dict[key_old].shape:
            if any([k in key_old for k in reshape_keys]):
                new_shape = model_state_dict[key].shape
                logger.warning('Reshaping {} -> {}. \n'.format(key_old, key))
                model_state_dict[key] = resize_2d(loaded_state_dict[key_old], new_shape)
            elif use_weightmap and 'cls_logits' in key:
                coco_in_objects365_inds = [
                    227, 26, 55, 202, 2, 44, 338, 346, 32, 336, 118, 299, 218,
                    25, 361, 59, 95, 161, 278, 82, 110, 22, 364, 134, 9, 350,
                    152, 323, 304, 130, 285, 289, 16, 172, 17, 18, 283, 305,
                    321, 35, 362, 88, 127, 174, 292, 37, 11, 6, 267, 212, 41,
                    58, 162, 237, 98, 48, 63, 81, 247, 23, 94, 326, 349, 178,
                    203, 259, 171, 60, 198, 213, 325, 282, 258, 33, 71, 353,
                    273, 318, 148, 330
                ]
                logger.info("Use coco_in_objects365_inds labelmap for COCO detection because of size mis-match, "
                      "Reshaping {} -> {}. \n".format(key_old, key))
                new_shape = model_state_dict[key].shape
                assert new_shape[0] == len(coco_in_objects365_inds)
                weight_inds_old = torch.as_tensor(coco_in_objects365_inds).to(loaded_state_dict[key_old].device)
                model_state_dict[key] = loaded_state_dict[key_old][weight_inds_old].to(model_state_dict[key].device)
            else:
                logger.info('Skip due to size mismatch: {} -> {}. \n'.format(key_old, key))
                continue
        else:
            model_state_dict[key] = loaded_state_dict[key_old]
        matched_keys.append(key)
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )
    missing_keys = set(current_keys)-set(matched_keys)
    if len(missing_keys):
        groups = _group_checkpoint_keys(missing_keys)
        msg_per_group = sorted(k + _group_to_str(v) for k, v in groups.items())
        msg = '\n'.join(sorted(msg_per_group))
        logger.warning('Some layers unloaded with pre-trained weight: \n' + msg)

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "", 1)] = value
    return stripped_state_dict

def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)

    # use strict loading
    model.load_state_dict(model_state_dict)

def _group_checkpoint_keys(keys):
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups

def _group_to_str(group):
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(sorted(group)) + "}"