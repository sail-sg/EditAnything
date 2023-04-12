from copy import deepcopy
import numpy as np
import torch
from torch import nn

# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()
        self.cfg = cfg
        self.bert_name = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT)

        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS

    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask,
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]
            features = None
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * mask.unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        else:
            # without padding, only consider positive_tokens
            max_len = (input != 0).sum(1).max().item()
            outputs = self.model(
                input_ids=input[:, :max_len],
                attention_mask=mask[:, :max_len],
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]

            features = None
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)
            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }
        return ret
