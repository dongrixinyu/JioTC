# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: focal_loss.py
# time: 2020-07-15 13:47

import pdb
import typing

import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=False):
        """
        focal_loss 损失函数, - (α * (1 - yi) ** γ) * ce_loss(xi, yi)
        步骤详细的实现了 focal_loss 损失函数.
        
        Args:
            alpha: 阿尔法α, 类别权重. 当α是列表时,为各类别权重, 当α为常数时, 
                类别权重为[α, 1-α, 1-α, ....], 常用于目标检测算法中抑制背景类
            gamma: 伽马γ, 难易样本调节参数.
            num_classes: 类别数量
            size_average: 损失计算方式,默认取均值
            
        Examples:
            >>> 
            >>> 
            
        """
        
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        
        if isinstance(alpha, list):
            # α 可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            assert len(alpha) == num_classes
            print('Focal_loss alpha = {}, 将对每一类权重进行精细化赋值'.format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            # 如果 α 为一个常数,则降低第一类的影响,在目标检测中为第一类
            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            assert alpha < 1
            print('Focal_loss alpha = {}, 将对背景类进行衰减'.format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
            
        self.gamma = gamma
        
    def forward(self, preds, labels):
        """
        focal_loss 损失计算
        
        Args:
            preds: 预测类别. size:[B,N,C] or [B,C]. 分别对应与检测与分类任务, B 批次, N检测框数, C类别数
            labels: 实际类别. size:[B,N] or [B]
        
        Return:
            loss: 标量值
            
        """
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        
        preds_softmax = F.softmax(preds, dim=1)
        # 这里并没有直接使用 log_softmax, 因为后面会用到 softmax 的结果
        preds_logsoft = torch.log(preds_softmax)
        
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        
        loss = - torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        
        loss = torch.mul(alpha, loss.t())
        
        tmp_num = 10
        #for i, k, j in zip(labels[:tmp_num], alpha[:tmp_num], loss[0][:tmp_num]):
        #    print(i.cpu().detach().numpy(), '\t', k.cpu().detach().numpy(), '\t', j.cpu().detach().numpy())

        #pdb.set_trace()
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
            
        return loss


