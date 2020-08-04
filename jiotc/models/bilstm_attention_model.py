# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: bilstm_attention_model.py
# time: 2020-07-15 17:37
# --------------------------------------------------------------------------------
'''
DESCRIPTION:
    1、word embedding：词向量一般是普通模型中参数量最大的部分，直接会导致模型的过拟合，
        因此，需要对 embedding 层的参数做 dropout，随机地在每次的计算中，mask 掉一部分
        词汇的 embedding，使之不参与计算，从而达到防止过拟合的效果。mask 是在 embedding
        weight matrix 矩阵中直接进行 mask，而非在序列寻找对应向量之后。
    2、rnn dropout：参考论文 A Theoretically Grounded Application of Dropout in 
        Recurrent Neural Networks。该方法实现在 tensorflow 框架中，但未在 pytorch 中
        实现。当前模型未  实现。
    3、dropout 的实现应采用 F.dropout2d 来完成，而非 nn.Dropout2d，因为无法避免在模型
        非训练阶段依然执行 dropout 的问题。
    4、attention 在并行计算时，应当先对句子进行截断，避免 pad 标签也进行 att 计算。但是
        即便使用 pad 进行计算，往往 attention score 会训练至，仅给 pad 一个非常小的值。
        句子越短，远小于 max_seq_len，attention 的计算情况越不理想。
        但此情况仅在数据非常充分时才能做到，因此需要在计算 att 前，对句子的 pad 进行剪除
        改造之后，普遍 F1 值提升 1~2%。
    

'''
import os
import pdb
import copy
import logging
from typing import Union, Optional, Dict, Any, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from jiotc.embeddings.base_embedding import BaseEmbedding
from .base_model import BaseModel


# Bidirectional LSTM neural network (many-to-one)
class BiLSTMAttentionModel(BaseModel):
    
    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_lstm': {
                'hidden_size': 128,
                'num_layers': 1,
                'dropout': 0.2,  # 当 num_layers == 1 时失效
                'bidirectional': True
            },
            'layer_dense': {
                'activation': 'softmax'
            }
        }
    
    def __init__(self, embed_model: Optional[BaseEmbedding] = None, 
                 device: Union['cuda', 'cpu'] = None,
                 hyper_parameters: Optional[Dict[str, Dict[str, Any]]] = None):
        ''' 
        self.device
        self.embedding_layer
        self.embedding
        self.embedding_size
        self.num_classes
        参数已知，可以直接使用
        '''
        super(BiLSTMAttentionModel, self).__init__(embed_model, device=device)
        
        self.hidden_size = hyper_parameters['layer_bi_lstm']['hidden_size']
        self.num_layers = hyper_parameters['layer_bi_lstm']['num_layers']
        self.dropout = hyper_parameters['layer_bi_lstm']['dropout']
        
        self.lstm = nn.LSTM(
            self.embedding_size, self.hidden_size, self.num_layers, 
            batch_first=True, bidirectional=True)
        # dropout 在 lstm 仅一层时不起作用
        
        self.w_attention = nn.Parameter(torch.Tensor(
            self.hidden_size * 2, self.hidden_size * 2))
        self.query_attention = nn.Parameter(torch.Tensor(self.hidden_size * 2, 1))
        
        self.fc = nn.Linear(self.hidden_size * 2,
                            self.num_classes)  # 2 for bidirection
        
        nn.init.uniform_(self.w_attention, -0.1, 0.1)
        nn.init.uniform_(self.query_attention, -0.1, 0.1)

    def forward(self, samples):
        
        masks = samples.gt(0)
        
        embeds = self._compute_embedding(samples)
        lstm_out = self._compute_lstm(embeds, masks)
        att_lstm_out_sum = self._compute_attention(lstm_out, masks)
        
        output = self.fc(att_lstm_out_sum)
        
        return output

    def _compute_embedding(self, samples):
        ''' 找出 samples 对应的 embedding，并在 Embedding 中随机 mask 掉一些词汇 '''
        if self.training:
            # 对 embedding weight matrix 做 dropout
            complete_embedding_weight = copy.deepcopy(self.embedding_layer.weight)
            embedding_weight = F.dropout2d(
                torch.unsqueeze(self.embedding_layer.weight, 0),
                p=self.dropout, training=self.training)[0]
            self.embedding_layer.weight.data.copy_(embedding_weight)
            
        embeds = self.embedding_layer(samples)
        
        if self.training:
            # 将原本未 dropout 参数还原
            self.embedding_layer.weight.data.copy_(complete_embedding_weight)
            
        #pdb.set_trace()
        return embeds
    
    def _compute_lstm(self, embeds, masks):
        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]  # 重新排序
        
        pack_sequence = pack_padded_sequence(
            embeds, lengths=sorted_seq_length, batch_first=True)
        
        # Forward propagate LSTM
        packed_output, _ = self.lstm(pack_sequence)
        # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        
        lstm_out = lstm_out[unperm_idx, :]
        return lstm_out
    
    def _compute_attention(self, lstm_out, masks):
        ''' 计算注意力权重，并对各时间片特征量加和 '''
        # dropout_layer
        lstm_out = lstm_out.permute(1, 0, 2)
        # [batch_size, seq_len, hidden_size] => [seq_len, batch_size, hidden_size * 2]
        #lstm_out = F.dropout2d(lstm_out, p=self.dropout, training=self.training)
        
        # attention_layer
        u = torch.tanh(torch.matmul(lstm_out, self.w_attention))
        # [seq_len, batch_size, hidden_size] => [seq_len, batch_size, 1]
        att = torch.matmul(u, self.query_attention)  # [seq_len, batch_size, 1]
        att = torch.squeeze(att, dim=-1)  # [seq_len, batch_size, 1] => [seq_len, batch_size]
        
        # att_score 各项加和为1，保证输出的数据分布均匀
        cur_seq_len, batch_size = att.shape
        cur_seq_masks = masks[:, :cur_seq_len].T  # 当前 mask,可能小于模型允许的最大长度，因此对齐
        att_without_pad = torch.where(cur_seq_masks == False, torch.ones_like(att) * (- float('inf')), att)
        
        att_score = F.softmax(att_without_pad, dim=0)  # [seq_len, batch_size]，应当对每个句子做截断，然后计算注意力权重
        
        if not self.training:
            # 观察实例
            #print(samples[:,0])
            #print([float(item) for item in att_score[:,0].cpu().detach().numpy()][:30])
            #pdb.set_trace()
            pass
            
        seq_len, batch_size = att_score.shape
        att_score = torch.unsqueeze(att_score, 2).expand(seq_len, batch_size, self.hidden_size * 2)
        # [seq_len, batch_size] => [seq_len, batch_size, hidden_size * 2]
        
        att_lstm_out = lstm_out * att_score  # 对应项点乘  [seq_len, batch_size, hidden_size * 2]
        att_lstm_out = att_lstm_out.permute(1, 0, 2)
        # [seq_len, batch_size, hidden_size * 2] => [batch_size, seq_len, hidden_size * 2]
        
        att_lstm_out_sum = torch.sum(att_lstm_out, dim=1)
        #pdb.set_trace()
        return att_lstm_out_sum
    
    
    
    
    
    
    
    