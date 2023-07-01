from gc import unfreeze
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from collections import OrderedDict
from transformers import BertModel, RobertaModel

class Encoder(nn.Module):
    def __init__(self, **kw):
        """
        Encoder that encodes raw sentence ids into token-level vectors (for BERT)
        """
        super(Encoder, self).__init__()

    def set_finetune(self, finetune):
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

class BERTEncoder(Encoder):
    def __init__(self, **kw):
        super(BERTEncoder, self).__init__(**kw)
        bert_dir = kw.get("bert_dir", "../data/pretrained_lm/bert")

        self.model = BertModel.from_pretrained("bert-base-uncased", cache_dir=bert_dir)

    def forward(self, x1, x2, mask1=None, mask2=None):
        bsz = x1.size(0)
        x1_len, x2_len = x1.size(1), x2.size(1)
        
        x = torch.cat([x1, x2], dim=1)
        
        type_ids = torch.empty_like(x)
        type_ids[:, :x1_len].data.fill_(0)
        type_ids[:, x1_len:].data.fill_(1)
        if mask1 is not None and mask2 is not None:
            mask = torch.cat([mask1, mask2], dim=1)
        else:
            mask = None

        x_output = self.model(x, attention_mask=mask, token_type_ids=type_ids)[0]

        return x_output, mask1, mask2

    def get_output_dim(self):
        return 768
    
class ROBERTAEncoder(Encoder):
    def __init__(self, **kw):
        super(ROBERTAEncoder, self).__init__(**kw)
        roberta_dir = kw.get("roberta_dir", "../data/pretrained_lm/roberta")

        self.model = RobertaModel.from_pretrained("roberta-base", cache_dir=roberta_dir)
        
                
    def forward(self, x1, x2, mask1=None, mask2=None):
        
        x = torch.cat([x1, x2], dim=1)
        if mask1 is not None and mask2 is not None:
            mask = torch.cat([mask1, mask2], dim=1)
        else:
            mask = None

        x_output = self.model(x, attention_mask=mask)[0]
        return x_output, mask1, mask2

    def get_output_dim(self):
        return 768
