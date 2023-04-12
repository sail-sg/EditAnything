from collections import OrderedDict
from torch import nn

from maskrcnn_benchmark.modeling import registry
from . import bert_model
from . import rnn_model
from . import clip_model


@registry.LANGUAGE_BACKBONES.register("bert-base-uncased")
def build_bert_backbone(cfg):
    body = bert_model.BertEncoder(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.LANGUAGE_BACKBONES.register("roberta-base")
def build_bert_backbone(cfg):
    body = bert_model.BertEncoder(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.LANGUAGE_BACKBONES.register("rnn")
def build_rnn_backbone(cfg):
    body = rnn_model.RNNEnoder(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


@registry.LANGUAGE_BACKBONES.register("clip")
def build_clip_backbone(cfg):
    body = clip_model.CLIPTransformer(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_backbone(cfg):
    assert cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE in registry.LANGUAGE_BACKBONES, \
        "cfg.MODEL.LANGUAGE_BACKBONE.TYPE: {} is not registered in registry".format(
            cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        )
    return registry.LANGUAGE_BACKBONES[cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE](cfg)
