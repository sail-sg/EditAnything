import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict

from .inference import make_atss_postprocessor
from .loss import make_atss_loss_evaluator
from .anchor_generator import make_anchor_generator_complex

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.layers import Scale, DYReLU, SELayer, ModulatedDeformConv
from maskrcnn_benchmark.layers import NaiveSyncBatchNorm2d, FrozenBatchNorm2d
from maskrcnn_benchmark.modeling.backbone.fbnet import *
from maskrcnn_benchmark.engine.inference import create_positive_map_label_to_token_from_positive_map
from ..utils import cat, concat_box_prediction_layers, permute_and_flatten

from maskrcnn_benchmark.utils.fuse_helper import FeatureResizer, func_attention, _make_mlp, _make_conv, _make_coord, \
    BiAttentionBlock, AttentionT2I, BiAttentionBlockForCheckpoint, BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertConfig, BertAttention, BertIntermediate, BertOutput, \
    BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
import torch.utils.checkpoint as checkpoint
import pdb

from maskrcnn_benchmark.modeling.language_backbone.clip_model import QuickGELU, LayerNorm, DropPath
from timm.models.layers import DropPath, trunc_normal_

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

        return pred_boxes


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 deformable=False,
                 bn_type=None):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                            groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(out_channels)
        elif bn_type == "sbn":
            bn_op = nn.SyncBatchNorm(out_channels)
        elif bn_type == "nsbn":
            bn_op = NaiveSyncBatchNorm2d(out_channels)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(out_channels)
        if bn_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyConv(torch.nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=nn.Conv2d,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_deform=False
                 ):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu:
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_deform:
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs):
        visual_feats = inputs["visual"]
        language_dict_features = inputs["lang"]

        next_x = []
        for level, feature in enumerate(visual_feats):

            conv_args = dict()
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1], **conv_args))
            if level < len(visual_feats) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](visual_feats[level + 1], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False)

            if self.AttnConv is not None:
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]

        features_dict = {"visual": next_x,
                         "lang": language_dict_features}

        return features_dict


class BertEncoderLayer(BertPreTrainedModel):
    def __init__(self, config,  clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        from maskrcnn_benchmark.modeling.rpn.modeling_bert import BertAttention, BertIntermediate, BertOutput

        self.attention = BertAttention(config,  clamp_min_for_underflow, clamp_max_for_overflow)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        hidden_states = language_dict_features["hidden"]
        attention_mask = language_dict_features["masks"]

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        self_attention_outputs = self.attention(
            hidden_states,
            extended_attention_mask,
            None,
            output_attentions=False,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        hidden_states = outputs[0]

        language_dict_features["hidden"] = hidden_states

        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features
                         }

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CLIPTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = self.config.MODEL.CLIP.WIDTH
        n_head = self.config.MODEL.CLIP.HEADS
        drop_path = self.config.MODEL.CLIP.DROP_PATH
        self.context_length = self.config.MODEL.CLIP.CONTEXT_LENGTH
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        x = language_dict_features["hidden"]
        mask = language_dict_features["masks"]
        # get extended attention mask for nn.MultiHeadAttention
        key_padding_mask = (1.0 - mask).to(torch.bool)

        x = x.permute(1, 0, 2)
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.permute(1, 0, 2)

        language_dict_features["hidden"] = x
        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features
                         }
        return features_dict


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self, cfg):
        super(VLFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg

        self.use_checkpoint = False
        if hasattr(cfg.MODEL.DYHEAD, 'USE_CHECKPOINT'):
            self.use_checkpoint = cfg.MODEL.DYHEAD.USE_CHECKPOINT
            self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # early fusion module
        print("EARLY FUSION ON, USING {}".format(cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE))
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            # single-direction (text->image)
            # text -> image
            self.t2i_attn = AttentionT2I(q_dim=self.joint_embedding_size,
                                           k_dim=self.lang_dim,
                                           embed_dim=self.embed_dim,
                                           num_heads=self.n_head,
                                           hidden_dim=self.t2i_hidden_dim,
                                           dropout=0.1,
                                           drop_path=.0,
                                           init_values=1.0 / cfg.MODEL.DYHEAD.NUM_CONVS,
                                           mode="t2i",
                                           use_layer_scale=cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE,
                                           clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW,
                                           clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW
                                           )

        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B":
            # bi-direction (text->image, image->text)
            self.b_attn = BiAttentionBlockForCheckpoint(v_dim=self.joint_embedding_size,
                        l_dim=self.lang_dim,
                        embed_dim=self.embed_dim,
                        num_heads=self.n_head,
                        hidden_dim=self.i2t_hidden_dim,
                        dropout=0.1,
                        drop_path=.0,
                        init_values=1.0 / cfg.MODEL.DYHEAD.NUM_CONVS,
                        cfg=cfg
                        )
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(self.lang_dim * 5,
                                self.lang_dim, 0.1)


        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            # single-direction (text->image)
            self.mapping_lang = _make_mlp(self.lang_dim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.joint_fusion = nn.ModuleList([_make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
                                               for _ in range(5)])

        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            # single-direction (text->image)
            self.mapping_lang = _make_mlp(self.lang_dim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.gamma = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(5))
            self.beta = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(5))

            self.joint_fusion = nn.ModuleList([_make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
                                               for _ in range(5)])

        else:
            print("NO FUSION INVOLVED.")

    def init_configs(self, cfg):
        # common params
        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        self.joint_mlp_layers = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS

        self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.n_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        self.coord_dim = 8
        self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE

        # mha params
        self.n_head = 8
        self.embed_dim = 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

    def forward(self, x):
        visual_features = x["visual"]
        language_dict_features = x["lang"]

        batch_size = visual_features[0].shape[0]
        device = visual_features[0].device

        fused_visual_features = None
        fused_language_dict_features = None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            language_feature = language_dict_features['hidden']
            mask = language_dict_features['masks']
            # text -> image
            if self.use_checkpoint:
                q0, q1, q2, q3, q4 = checkpoint.checkpoint(
                    self.t2i_attn,
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_feature, language_feature,
                    mask,
                    self.dummy_tensor
                )
            else:
                q0, q1, q2, q3, q4 = self.t2i_attn(
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_feature, language_feature,
                    attention_mask=mask
                )

            fused_visual_features = [q0, q1, q2, q3, q4]
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B":
            if self.use_checkpoint:
                q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = checkpoint.checkpoint(self.b_attn,
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_dict_features['hidden'],
                    language_dict_features['masks'],
                    self.dummy_tensor
                )
            else:
                q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = self.b_attn(
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_dict_features['hidden'],
                    language_dict_features['masks'],
                    self.dummy_tensor
                )

            fused_visual_features = [q0, q1, q2, q3, q4]
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                language_features = self.shrink_lang(torch.cat([l0, l1, l2, l3, l4], dim = -1))
            else:
                language_features = l0

            language_dict_features['hidden'] = language_features
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            # text -> image
            language_feature = language_dict_features['aggregate']
            language_feature = self.mapping_lang(language_feature)
            visu_feat = []
            for ii, feat in enumerate(visual_features):
                attn_feat = func_attention(feat, language_feature, smooth=1, raw_feature_norm="softmax")
                visu_feat.append(attn_feat)

            fused_visual_features = [fusion(feat) for feat, fusion in zip(visu_feat, self.joint_fusion)]
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            # text -> image
            # relative position embedding
            coord_feats = [_make_coord(batch_size, x.shape[2], x.shape[3]) for x in visual_features]
            # I only use a global representation of language
            # you can also use more complex modeling using word-level representations
            # Usage: lang_feat = lang_feat['words'] shape [seq_len, dim]
            language_feature = language_dict_features['aggregate']
            language_feature = self.mapping_lang(language_feature)

            # attention mechanism for fusion
            gamma = [F.tanh(gamma(language_feature)) for gamma in self.gamma]
            beta = [F.tanh(beta(language_feature)) for beta in self.beta]

            visu_feat = []
            for ii, feat in enumerate(visual_features):
                coord_feat = coord_feats[ii].to(device)
                feat = torch.cat([feat, coord_feat], dim=1)
                b = beta[ii].view(batch_size, -1, 1, 1).expand_as(feat)
                g = gamma[ii].view(batch_size, -1, 1, 1).expand_as(feat)
                feat = F.relu(g * feat + b)
                visu_feat.append(feat)

            fused_visual_features = [fusion(feat) for feat, fusion in zip(visu_feat, self.joint_fusion)]
            fused_language_dict_features = language_dict_features

        else:
            fused_visual_features = visual_features
            fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features}

        return features_dict


class VLDyHead(torch.nn.Module):
    def __init__(self, cfg):
        super(VLDyHead, self).__init__()
        self.cfg = cfg
        # bert_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
            lang_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        elif cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
            lang_cfg = cfg
        else:
            lang_cfg = None
            raise NotImplementedError

        num_classes = cfg.MODEL.DYHEAD.NUM_CLASSES - 1
        num_tokens = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS

        if cfg.MODEL.DYHEAD.USE_GN:
            bn_type = ['gn', cfg.MODEL.GROUP_NORM.NUM_GROUPS]
        elif cfg.MODEL.DYHEAD.USE_NSYNCBN:
            bn_type = 'nsbn'
        elif cfg.MODEL.DYHEAD.USE_SYNCBN:
            bn_type = 'sbn'
        else:
            bn_type = None

        use_dyrelu = cfg.MODEL.DYHEAD.USE_DYRELU
        use_dyfuse = cfg.MODEL.DYHEAD.USE_DYFUSE
        use_deform = cfg.MODEL.DYHEAD.USE_DFCONV

        if cfg.MODEL.DYHEAD.CONV_FUNC:
            conv_func = lambda i, o, s: eval(cfg.MODEL.DYHEAD.CONV_FUNC)(i, o, s, bn_type=bn_type)
        else:
            conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON:
                # cross-modality fusion
                dyhead_tower.append(
                    VLFuse(cfg)
                )
                # self language path
                if i < cfg.MODEL.DYHEAD.NUM_CONVS - 1 or cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
                    # dyhead_tower.append(
                    #     BertEncoderLayer(
                    #     bert_cfg,
                    #     clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                    #     clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                    # )
                    if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
                        dyhead_tower.append(
                            BertEncoderLayer(
                                lang_cfg,
                                clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                                clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                        )
                    elif cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
                        dyhead_tower.append(
                            CLIPTransformerLayer(lang_cfg)
                        )
                    else:
                        raise NotImplementedError

                else:
                    dyhead_tower.append(
                        DummyLayer()
                    )

            # self vision path
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and in_channels == channels) if i == 0 else use_dyrelu,
                    use_dyfuse=(use_dyfuse and in_channels == channels) if i == 0 else use_dyfuse,
                    use_deform=(use_deform and in_channels == channels) if i == 0 else use_deform,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        log_scale = self.cfg.MODEL.DYHEAD.LOG_SCALE

        # soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            self.token_logits = nn.Conv2d(channels, num_anchors * num_tokens, kernel_size=1)
            # ABLATION
            # self.token_logits = nn.Conv2d(channels, num_anchors * num_tokens, kernel_size=1, bias=False)
            # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
            # self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # contrastive alignment head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS == False
            contrastive_hdim = cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM
            self.contrastive_align_projection_image = nn.Conv2d(channels, num_anchors * contrastive_hdim, kernel_size=1)
            self.contrastive_align_projection_text = nn.Linear(channels, contrastive_hdim, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)

        # dot product soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS == False
            self.dot_product_projection_image = nn.Identity()
            self.dot_product_projection_text = nn.Linear(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                         num_anchors * channels, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
            # DEBUG
            # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
            self.bias_lang = nn.Parameter(torch.zeros(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # if use soft token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            for modules in [self.token_logits]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

            torch.nn.init.constant_(self.token_logits.bias, bias_value)
            # print(torch.norm(self.token_logits.weight))

        # if use contrastive loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            for modules in [self.contrastive_align_projection_image]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        # if use dot product token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            for modules in [self.dot_product_projection_image]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, bias_value)
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
                lang_cfg = BertConfig.from_pretrained("bert-base-uncased")
                lang_cfg.hidden_size = cfg.MODEL.CLIP.WIDTH
                lang_cfg.vocab_size = cfg.MODEL.CLIP.VOCAB_SIZE
            self.mlm_head = BertLMPredictionHead(
                lang_cfg
            ) #nn.Linear(hidden_size, config.vocab_size, bias=False)

    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None):
        logits = []
        bbox_reg = []
        centerness = []

        feat_inputs = {"visual": x,
                       "lang": language_dict_features}

        dyhead_tower = self.dyhead_tower(feat_inputs)

        # soft token
        t_logits = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            t_logits = []
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
            embedding = dyhead_tower["lang"]["hidden"]
        
        # MLM loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            mlm_logits = self.mlm_head(embedding)
        else:
            mlm_logits = None

        # contrastive
        contrastive_logits = None
        proj_tokens = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            contrastive_logits = []
            # follow MDETR's way
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(embedding), p=2, dim=-1
            )

        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            dot_product_logits = []
            # norm
            embedding = F.normalize(embedding, p=2, dim=-1)
            dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
            # w/o norm
            # dot_product_proj_tokens = self.dot_product_projection_text(embedding / 28.0)

            dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0

        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats = None
        shallow_text_emb = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS \
                or self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            shallow_img_emb_feats = []
            shallow_text_emb = embedding

        # print([v.shape for v in x])
        # shallow contrastive: use the feature from swint backbone
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            for b, feature in enumerate(swint_feature_c4):
                # BF, CF, HF, WF = feat.shape
                # shallow_img_emb = permute_and_flatten(feat, BF, -1, CF, HF, WF)
                shallow_img_emb_feats.append(feature)

        fused_visual_features = None
        if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
            fused_visual_features = []

        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
                t_logits.append(self.token_logits(dyhead_tower["visual"][l]))

                # ABLATION
                # b = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                # x = dyhead_tower["visual"][l]
                # B, C, H, W = x.shape
                # bias = b.repeat(B, 1, H, W)
                # t_logits.append(self.token_logits(dyhead_tower["visual"][l] + bias) + self.bias0)

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
                x = dyhead_tower["visual"][l]
                B, _, H, W = x.shape
                C = proj_tokens.shape[2]
                proj_queries = self.contrastive_align_projection_image(dyhead_tower["visual"][l])
                proj_queries = permute_and_flatten(proj_queries, B, -1, C, H, W)
                normalized_img_emb = F.normalize(proj_queries, p=2, dim=-1)
                normalized_text_emb = proj_tokens
                contrastive_logit = (
                        torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.log_scale.exp())
                contrastive_logits.append(contrastive_logit)

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
                x = dyhead_tower["visual"][l]
                if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                    fused_visual_features.append(x)
                B, C, H, W = x.shape

                # add bias (language)
                dot_product_proj_queries = self.dot_product_projection_image(x)
                dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)

                A = dot_product_proj_queries.shape[1]
                bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)

                dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
                if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                    dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                    dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
                dot_product_logits.append(dot_product_logit)

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS:
                feat = feature
                BF, CF, HF, WF = feat.shape
                shallow_img_emb = permute_and_flatten(feat, BF, -1, CF, HF, WF)
                shallow_img_emb_feats.append(shallow_img_emb)

        # no matter the feature is from backboone or from fpn, we use shallow_img_embs all the time
        if shallow_img_emb_feats is not None and shallow_text_emb is not None:
            # shallow_img_embs = torch.cat(shallow_img_embs, dim=1)
            proj_tokens = shallow_text_emb
        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features


class VLDyHeadModule(torch.nn.Module):

    def __init__(self, cfg):
        super(VLDyHeadModule, self).__init__()
        self.cfg = cfg
        self.head = VLDyHead(cfg)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder)
        self.box_selector_train = make_atss_postprocessor(cfg, box_coder, is_train=True)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder, is_train=False)
        self.anchor_generator = make_anchor_generator_complex(cfg)

        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            self.resizer = FeatureResizer(
                input_feat_size=self.lang_dim,
                output_feat_size=self.joint_embedding_size,
                dropout=self.joint_embedding_dropout
            )
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
            self.tunable_linear = torch.nn.Linear(self.lang_dim, 1000, bias=False)
            self.tunable_linear.weight.data.fill_(0.0)

    def forward(self, images, features, targets=None,
                language_dict_features=None,
                positive_map=None,
                captions=None,
                swint_feature_c4=None
                ):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            # resizer needed
            embedding = language_dict_features['embedded']
            embedding = self.resizer(embedding)
        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # no resizer needed
            embedding = language_dict_features['embedded']
        else:
            embedding = None

        if "masks" in language_dict_features:
            text_masks = language_dict_features["masks"]
        else:
            text_masks = None
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
            embedding = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + embedding
            language_dict_features['embedded'] = embedding
            language_dict_features['hidden'] = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + language_dict_features['hidden']

        box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features = self.head(features,
                                                                        language_dict_features,
                                                                        embedding,
                                                                        swint_feature_c4
                                                                        )
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors,
                                       captions,
                                       positive_map,
                                       token_logits,
                                       proj_tokens,
                                       contrastive_logits,
                                       dot_product_logits,
                                       text_masks,
                                       mlm_logits = mlm_logits,
                                       mlm_labels = language_dict_features["mlm_labels"],
                                       shallow_img_emb_feats=shallow_img_emb_feats,
                                       fused_visual_features=fused_visual_features
                                       )
        else:
            return self._forward_test(box_regression, centerness, anchors,
                                      box_cls,
                                      token_logits,
                                      dot_product_logits,
                                      positive_map,
                                      fused_visual_features=fused_visual_features
                                      )

    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors,
                       captions=None,
                       positive_map=None,
                       token_logits=None,
                       proj_tokens=None,
                       contrastive_logits=None,
                       dot_product_logits=None,
                       text_masks=None,
                       mlm_logits=None,
                       mlm_labels=None,
                       shallow_img_emb_feats=None,
                       fused_visual_features=None
                       ):

        loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, loss_shallow_contrastive = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors,
            captions,
            positive_map,
            token_logits,
            proj_tokens,
            contrastive_logits,
            dot_product_logits,
            text_masks,
            shallow_img_emb_feats
        )

        losses = {
            # "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

        if mlm_labels is not None and mlm_logits is not None:
            losses["mlm_loss"] = nn.CrossEntropyLoss(ignore_index = -100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)) * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS:
            losses["loss_cls"] = loss_box_cls
        else:
            losses["loss_cls"] = 0.0 * loss_box_cls

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            losses["loss_token"] = loss_token * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            losses["loss_contrastive_align"] = loss_contrastive_align * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            losses["loss_dot_product_token"] = loss_dot_product_token * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS or \
                self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            losses["loss_shallow_contrastive"] = loss_shallow_contrastive * \
                                                 self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT

        if self.cfg.MODEL.RPN_ONLY:
            return None, losses, None
        else:
            # Let's just use one image per batch
            assert (box_regression[0].shape[0]) == 1
            positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=1)
            boxes = self.box_selector_train(box_regression, centerness, anchors,
                                        box_cls,
                                        token_logits,
                                        dot_product_logits,
                                        positive_map=positive_map_label_to_token
                                        )
            train_boxes = []
            for b, t in zip(boxes, targets):
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                train_boxes.append(cat_boxlist([b, tb]))
            return train_boxes, losses, fused_visual_features

    def _forward_test(self, box_regression, centerness, anchors,
                      box_cls=None,
                      token_logits=None,
                      dot_product_logits=None,
                      positive_map=None,
                      fused_visual_features=None
                      ):

        boxes = self.box_selector_test(box_regression, centerness, anchors,
                                       box_cls,
                                       token_logits,
                                       dot_product_logits,
                                       positive_map,
                                       )
        return boxes, {}, fused_visual_features
