from collections import OrderedDict
import logging
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from maskrcnn_benchmark.config import try_to_find

from timm.models.layers import DropPath, trunc_normal_

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class CLIPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.use_checkpoint = cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
        print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT)

        self.context_length = self.cfg.MODEL.CLIP.CONTEXT_LENGTH
        self.width = self.cfg.MODEL.CLIP.WIDTH
        self.layers = self.cfg.MODEL.CLIP.LAYERS
        self.heads = self.cfg.MODEL.CLIP.HEADS
        self.drop_path = self.cfg.MODEL.CLIP.DROP_PATH
        self.vocab_size = self.cfg.MODEL.CLIP.VOCAB_SIZE

        self.token_embedding = nn.Embedding(self.vocab_size, self.width)

        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, self.width)
        )

        # attn_mask = self.build_attention_mask()
        attn_mask = None

        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.layers)]  # stochastic depth decay rule
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(self.width, self.heads, attn_mask, dpr[i])
                for i in range(self.layers)
            ]
        )

        self.ln_final = LayerNorm(self.width)

        trunc_normal_(self.positional_embedding, std=.02)
        # nn.init.normal_(self.token_embedding, std=.02)
        trunc_normal_(self.token_embedding.weight, std=.02)
        self.apply(self._init_weights)

        # loading pre-trained weight from our CLIP models
        if len(self.cfg.MODEL.LANGUAGE_BACKBONE.WEIGHT) > 0:
            self.init_weights(pretrained=try_to_find(self.cfg.MODEL.LANGUAGE_BACKBONE.WEIGHT),
                              pretrained_layers=['*'])

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)

    def resize_pos_embed_1d(self, posemb, shape_new):
        # rescale the grid of position embeddings when loading from state_dict
        ntok_old = posemb.shape[0]
        if ntok_old > 1:
            ntok_new = shape_new[0]
            posemb_grid = posemb.unsqueeze(dim=0).permute(0, 2, 1).unsqueeze(dim=-1)
            posemb_grid = F.interpolate(posemb_grid, size=[ntok_new, 1], mode='bilinear')
            posemb_grid = posemb_grid.squeeze(dim=-1).permute(0, 2, 1).squeeze(dim=0)
            posemb = posemb_grid
        return posemb

    def init_weights(self, pretrained="", pretrained_layers=[], verbose=False):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location="cpu")
            logger.info(f'=> loading pretrained clip text model {pretrained}')
            model_dict = self.state_dict()

            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if k.startswith('text.') and k[5:] in model_dict.keys():
                        need_init_state_dict[k[5:]] = v

            # notice the context length now changes from 77 to 256, so we need to resize the positional embedding
            if "positional_embedding" in need_init_state_dict.keys():
                old_pos_embed = need_init_state_dict["positional_embedding"].float()
                new_pos_embed = self.resize_pos_embed_1d(old_pos_embed,
                                                         (self.cfg.MODEL.CLIP.CONTEXT_LENGTH, old_pos_embed.shape[1]))
                need_init_state_dict["positional_embedding"] = new_pos_embed
            self.load_state_dict(need_init_state_dict, strict=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
        }

    def forward(self, text):
        input = text["input_ids"]
        mask = text["attention_mask"]
        # get extended attention mask for nn.MultiHeadAttention
        key_padding_mask = (1.0 - mask).to(torch.bool)

        x = self.token_embedding(input)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND

        for resblock in self.resblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(resblock, x, key_padding_mask)
            else:
                x = resblock(x, key_padding_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)

        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        ret = {
            "aggregate": x,
            "embedded": x,
            "masks": mask,
            "hidden": x
        }

        return ret