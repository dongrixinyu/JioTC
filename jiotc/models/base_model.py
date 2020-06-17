# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://eliyar.biz

# file: bare_embedding.py
# time: 2020-06-12 11:27

import os
import pdb
import logging
from typing import Union, Optional, Dict, Any, Tuple


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from text_classification.embeddings.base_embedding import BaseEmbedding


# Bidirectional LSTM neural network (many-to-one)
class BaseModel(nn.Module):
    
    def __init__(self, embed_model: Optional[BaseEmbedding] = None):
        
        super(BaseModel, self).__init__()
        
        self.embedding = embed_model
        
        if hyper_parameters['device'] is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        self.device = device
        
        # 构建 embedding 以及预处理时已经确定的参数，可以复制
        self.embedding_size = self.embedding.embedding_size
        self.num_classes = len(self.embedding.label2idx)

    def forward(self, samples):
        
        masks = samples.gt(0)
        embeds = self.embedding.embedding_layer(samples).to(self.device)
        
        # 按长短调整样本顺序
        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]
        
        return out


