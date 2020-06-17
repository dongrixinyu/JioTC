# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: bare_embedding.py
# time: 2020-06-12 11:27

import os
import pdb
import logging
from typing import Union, Optional, Dict, Any, Tuple


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from jiotc.embeddings.base_embedding import BaseEmbedding


# Bidirectional LSTM neural network (many-to-one)
class BaseModel(nn.Module):
    
    def __init__(self, embed_model: Optional[BaseEmbedding] = None,
                 device: Union['cuda', 'cpu'] = None):
        ''' load device and embedding layer and params '''
        super(BaseModel, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        self.device = device
        
        self.embedding = embed_model
        self.embedding._build_model()  # 加载 embed 层
        self.embedding_layer = self.embedding.embedding_layer.to(self.device)
        
        # 构建 embedding 以及预处理时已经确定的参数，可以复制
        self.embedding_size = self.embedding.embedding_size
        self.num_classes = len(self.embedding.label2idx)

    def forward(self, inputs_data):
        raise NotImplementedError


