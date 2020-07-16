# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: bare_embedding.py
# time: 2020-06-12 11:27

import os
import pdb
import logging
from typing import Union, Optional


import torch.nn as nn

from .base_embedding import BaseEmbedding
from jiotc.processor import Processor


class BareEmbedding(BaseEmbedding):

    """Embedding layer without pre-training, train embedding layer while training model"""

    def __init__(self,
                 sequence_length: Union[int, str] = 'auto',
                 embedding_size: int = 100,
                 processor: Optional[Processor] = None,
                 from_saved_model: bool = False):
        """
        Init bare embedding (embedding without pre-training)

        Args:
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
            embedding_size: Dimension of the dense embedding.
        """
        super(BareEmbedding, self).__init__(
            sequence_length=sequence_length,
            embedding_size=embedding_size,
            processor=processor,
            from_saved_model=from_saved_model)
        
        #if not from_saved_model:
        #    self._build_model()

    def _build_model(self, **kwargs):
        if self.sequence_length == 0 or \
                self.sequence_length == 'auto' or \
                self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            # 指定该向量层
            self.embedding_layer = nn.Embedding(self.token_count, self.embedding_size)



