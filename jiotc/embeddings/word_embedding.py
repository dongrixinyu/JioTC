# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://eliyar.biz

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
from typing import Union, Optional, Dict, Any, List, Tuple

import numpy as np
from gensim.models import KeyedVectors

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
                 w2v_path: str,
                 w2v_kwargs: Dict[str, Any] = None,
                 sequence_length: Union[Tuple[int, ...], str, int] = 'auto',
                 processor: Optional[Processor] = None,
                 from_saved_model: bool = False):
        """

        Args:
            w2v_path: word2vec file path
            w2v_kwargs: params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` -
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            sequence_length: ``'auto'``, ``'variable'`` or integer. When using ``'auto'``, use the 95% of corpus length
                as sequence length. When using ``'variable'``, model input shape will set to None, which can handle
                various length of input, it will use the length of max sequence in every batch for sequence length.
                If using an integer, let's say ``50``, the input output sequence length will set to 50.
            processor:
            
        """
        if w2v_kwargs is None:
            w2v_kwargs = {}
        self.w2v_path = w2v_path
        self.w2v_kwargs = w2v_kwargs
        self.w2v_model_loaded = False

        super(WordEmbedding, self).__init__(sequence_length=sequence_length,
                                            embedding_size=0,
                                            processor=processor,
                                            from_saved_model=from_saved_model)

        if not from_saved_model:
            self._build_token2idx_from_w2v()
            if self.sequence_length != 'auto':
                self._build_model()

    def _build_token2idx_from_w2v(self):
        # 加载
        w2v = KeyedVectors.load_word2vec_format(self.w2v_path, **self.w2v_kwargs)
        
        # 整理 token2idx
        token2idx = {
            self.processor.token_pad: 0,
            self.processor.token_unk: 1,
            self.processor.token_bos: 2,
            self.processor.token_eos: 3,
            self.processor.token_sap: 4
        }

        for token in w2v.index2word:
            token2idx[token] = len(token2idx)

        # 赋值
        vector_matrix = np.zeros((len(token2idx), w2v.vector_size))
        vector_matrix[1] = np.random.rand(w2v.vector_size)
        vector_matrix[4:] = w2v.vectors

        self.embedding_size = w2v.vector_size
        self.w2v_vector_matrix = vector_matrix
        self.w2v_token2idx = token2idx
        self.w2v_top_words = w2v.index2entity[:50]
        self.w2v_model_loaded = True

        self.processor.token2idx = self.w2v_token2idx
        self.processor.idx2token = dict([(value, key) for key, value in self.w2v_token2idx.items()])
        logging.debug('-' * 40)
        logging.debug('Loaded gensim word2vec model')
        logging.debug('model        : {}'.format(self.w2v_path))
        logging.debug('word count   : {}'.format(len(self.w2v_vector_matrix)))
        logging.debug('Top 50 word  : {}'.format(self.w2v_top_words))
        logging.debug('-' * 40)

    def _build_model(self, **kwargs):
        
        
        embedding = nn.Embedding.from_pretrained(weight)
        
        if self.token_count == 0:
            logging.debug('need to build after build_word2idx')
        else:
            input_tensor = L.Input(shape=(self.sequence_length,),
                                   name=f'input')
            layer_embedding = L.Embedding(self.token_count,
                                          self.embedding_size,
                                          weights=[self.w2v_vector_matrix],
                                          trainable=False,
                                          name=f'layer_embedding')

            embedded_tensor = layer_embedding(input_tensor)
            self.embed_model = keras.Model(input_tensor, embedded_tensor)

    def analyze_corpus(self,
                       x: Union[Tuple[List[List[str]], ...], List[List[str]]],
                       y: Union[List[List[Any]], List[Any]]):
        """
        Prepare embedding layer and pre-processor for labeling task

        Args:
            x:
            y:

        Returns:

        """
        if not self.w2v_model_loaded:
            self._build_token2idx_from_w2v()

        super(WordEmbedding, self).analyze_corpus(x, y)


if __name__ == "__main__":
    w2v = WordEmbedding()
    































