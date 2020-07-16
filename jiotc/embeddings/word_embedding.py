# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: word_embedding.py
# time: 2020-06-12 11:27

'''
如何训练、加载、使用一个词向量文件

# 训练
>>> from gensim.test.utils import common_texts
>>> from gensim.models import Word2Vec
>>> model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
>>> word_vectors = model.wv

# 保存
>>> word_vectors.save("vectors_wv")

# 加载
>>> from gensim.models import KeyedVectors
>>> word_vectors = KeyedVectors.load("vectors_wv", mmap='r')
>>> wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
>>> wv_from_bin = KeyedVectors.load_word2vec_format(datapath('word2vec_vector.bin'), binary=True)  # C text format

'''


import pdb
import logging
import random
from typing import Union, Optional, Dict, Any, List, Tuple


import numpy as np
import torch
import torch.nn as nn

from .base_embedding import BaseEmbedding
from jiotc.processor import Processor


class WordEmbedding(BaseEmbedding):
    """Pre-trained word2vec embedding"""

    def info(self):
        info = super(WordEmbedding, self).info()
        info['config'] = {
            'w2v_path': self.w2v_path,
            'w2v_kwargs': self.w2v_kwargs,
            'sequence_length': self.sequence_length
        }
        return info

    def __init__(self,
                 embedding_weight: Dict[str, List[float]],
                 sequence_length: Union[int, str] = 'auto',
                 processor: Optional[Processor] = None,
                 trainable_embedding: bool = True,
                 from_saved_model: bool = False):
        """

        Args:
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
            processor:
            
        """
        
        # 检查预训练词向量权重是否符合要求
        for idx, (token, embedding_list) in enumerate(embedding_weight.items()):
            if idx == 0:
                self.embedding_size = len(embedding_list)
            assert len(embedding_list) == self.embedding_size
        
        self.embedding_weight = embedding_weight
        self.trainable_embedding = trainable_embedding
        
        super(WordEmbedding, self).__init__(
            sequence_length=sequence_length,
            embedding_size=self.embedding_size,
            processor=processor,
            from_saved_model=from_saved_model)

        #pdb.set_trace()
        #if not from_saved_model:
        #    self._build_token2idx_from_w2v()
        #    if self.sequence_length != 'auto':
        #        self._build_model()

    def _build_model(self, **kwargs):
        
        self.embedding_layer = nn.Embedding(self.token_count, self.embedding_size)
        
        
        #pdb.set_trace()
        # 整理预训练词向量参数至 embedding layer
        embedding_weight = list()
        for token, idx in self.token2idx.items():
            if token in self.embedding_weight:
                embedding_weight.append(self.embedding_weight[token])
            else:
                tmp_token_embedding_weight = [
                    (random.random() - 0.5) * 2 for i in range(self.embedding_size)]
                embedding_weight.append(tmp_token_embedding_weight)
            
        embedding_weight = np.asarray(embedding_weight)
        
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.embedding_layer.weight.requires_grad = self.trainable_embedding
        


if __name__ == "__main__":
    w2v = WordEmbedding()
    



