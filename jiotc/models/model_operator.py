# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: bare_embedding.py
# time: 2020-06-12 11:27

import os
import pdb
import copy
import random
import logging
from typing import Union, Optional, Dict, Any


import torch
import torch.nn as nn
from torch.nn import functional as F

from jiotc.util import compute_f1_single_label, compute_accuracy


class ModelOperator(object):
    """ 分类模型的操作类 """
    
    @classmethod
    def get_training_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        #raise NotImplementedError
        return {
            'learning_rate': 1e-3,
            'epoch': 10,
            'batch_size': 128,
            'print_every': 100,
        }
    
    def __init__(self, torch_model=None,
                 dataset=None,
                 hyper_parameters: Optional[Dict[str, Dict[str, Any]]] = None):
                 #device: Union['cuda', 'cpu'] = None):
        # Device configuration
        #if device is None:
        #    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = device
        
        # 建立的模型
        self.torch_model = torch_model  #.to(device)
        
        self.training_hyper_parameters = hyper_parameters

    def compile_model(self, loss_func=None, optimizer=None, metric=['f1', 'acc'],
                      regularization='l2'):
        # 指定损失函数
        if loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = loss_func
        
        # 指定优化器
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.torch_model.parameters(),
                lr=self.training_hyper_parameters['learning_rate'],
                weight_decay=1e-3)
        else:
            self.optimizer = optimizer
            
        #if regularization == 'l2':
            
        
            
    def _dispatch_dataset(self, batch_size, dataset_x, dataset_y=None, 
                          pad_to_batch=False):
        ''' 按 batch_size 分配数据集 '''
        # 打乱数据集
        if dataset_y is None:
            dataset = dataset_x
        else:
            dataset = [[x, y] for x, y in zip(dataset_x, dataset_y)]
            random.shuffle(dataset)
        
        # 将数据集样本补齐至 batch_size 整数倍
        sample_num = len(dataset)
        if pad_to_batch:
            append_num = batch_size - sample_num % batch_size
            dataset.extend(copy.deepcopy(dataset[: append_num]))
            
        for i in range(0, len(dataset), batch_size):
            cur_batch = dataset[i: i+batch_size]
            if dataset_y is None:
                cur_batch_x = cur_batch
                yield cur_batch_x
                
            else:
                cur_batch_x = [item[0] for item in cur_batch]
                cur_batch_y = [item[1] for item in cur_batch]
                #pdb.set_trace()
                yield cur_batch_x, cur_batch_y
    
    @property
    def device(self):
        return self.torch_model.device
    
    def print_model_structure(self):
        print('-' * 30 + '\nModel structure: ')
        print(self.torch_model)
        print('-' * 30 + '\nParams structure: ')
        total_params_num = 0
        for name, parameters in self.torch_model.named_parameters():
            cur_num = 1
            for i in parameters.size():
                cur_num *= i
            print('\t', name, ':\t', list(parameters.size()),
                  '\t', format(cur_num, ','))
            total_params_num += cur_num
        print('Trainable params number: {:,}'.format(total_params_num))
    
    def train(self,
              train_dataset_x, train_dataset_y,
              valid_dataset_x, valid_dataset_y):
        ''' 训练一个模型 '''
        
        if self.torch_model is None:
            raise ValueError('`torch_model` should NOT be None.')
        
        self.print_model_structure()
        
        batch_size = self.training_hyper_parameters['batch_size']
        print_every = self.training_hyper_parameters['print_every']
        
        # Train the model
        total_step = int(len(train_dataset_x) / batch_size)
        
        for epoch in range(self.training_hyper_parameters['epoch']):
            for i, (inputs, labels) in enumerate(
                self._dispatch_dataset(batch_size, train_dataset_x,
                                       train_dataset_y, pad_to_batch=True)):
                
                self.torch_model.training = True
                
                inputs_tensor = self.torch_model.embedding.process_x_dataset(inputs)
                labels_tensor = self.torch_model.embedding.process_y_dataset(labels)
                
                inputs_tensor = torch.LongTensor(inputs_tensor).to(self.device)
                labels_tensor = torch.LongTensor(labels_tensor).to(self.device)

                # Forward pass
                outputs = self.torch_model(inputs_tensor)
                #pdb.set_trace()
                loss = self.loss_func(outputs.to(self.device),
                                      labels_tensor.to(self.device))
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) == total_step:
                    #compute_f1_single_label
                    valid_loss, valid_accuracy, valid_weighted_f1, valid_average_f1 = self.evaluate(
                        valid_dataset_x, valid_dataset_y, print_detail=False)
                    #pdb.set_trace()
                    train_loss, train_accuracy, train_weighted_f1, train_average_f1 = self.evaluate(
                        train_dataset_x, train_dataset_y, print_detail=False)
                    print('Epoch [{}/{}], Step [{}/{}], \n\tTrain Loss: {:.4f}, '
                          'Valid Loss: {:.4f}, \n\tTrain Micro F1: {:.2%}, '
                          'Valid Micro F1: {:.2%}, \n\tTrain Macro F1: {:.2%}, '
                          'Valid Macro F1: {:.2%}, \n\tTrain Accuracy: {:.2%}, '
                          'Valid Accuracy: {:.2%}.'
                          .format(epoch + 1, self.training_hyper_parameters['epoch'],
                                  i + 1, total_step, train_loss, valid_loss,
                                  train_weighted_f1, valid_weighted_f1, 
                                  train_average_f1, valid_average_f1,
                                  train_accuracy, valid_accuracy))
                    
                elif (i + 1) % print_every == 0:
                    
                    valid_loss, valid_accuracy, valid_weighted_f1, valid_average_f1 = self.evaluate(
                        valid_dataset_x, valid_dataset_y, print_detail=False)
                    #pdb.set_trace()
                    
                    print('Epoch [{}/{}], Step [{}/{}], \n\tTrain Loss: {:.4f}, '
                          'Valid Loss: {:.4f}, \n\tValid Micro F1: {:.2%}, '
                          'Valid Macro F1: {:.2%}, Valid Accuracy: {:.2%}'
                          .format(epoch + 1, self.training_hyper_parameters['epoch'],
                                  i + 1, total_step, loss.item(), valid_loss,
                                  valid_weighted_f1, valid_average_f1,
                                  valid_accuracy))

    def save(self, model_name: str = 'model.ckpt'):
        #torch.save(self.torch_model.state_dict(), 'model.ckpt')  # 仅保存参数
        torch.save(self.torch_model, model_name)
    
    def load(self, model_name: str = 'model.ckpt', device: str = None):
        # apply to the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        if os.path.isabs(model_name):
            model_path = model_name
        else:
            if model_name in os.listdir(os.path.abspath('.')):  # 当前路径下
                model_path = os.path.join(os.path.abspath('.'), model_name)
            else:
                raise ValueError('the model {} does not exist.'.format(model_name))
        
        # resnet.load_state_dict(torch.load('params.ckpt'))  # 仅加载模型参数，可以用于继续训练
        self.torch_model = torch.load(model_path)
        self.torch_model = self.torch_model.to(device)

    def evaluate(self, valid_dataset_x, valid_dataset_y, print_detail=True):
        
        if self.torch_model is None:
            raise ValueError('`torch_model` should NOT be None.')
        
        batch_size = self.training_hyper_parameters['batch_size']
        
        with torch.no_grad():
            self.torch_model.training = False
            pred_y_list = list()
            true_y_list = list()
            loss_list = list()
            for i, (inputs, labels) in enumerate(
                self._dispatch_dataset(batch_size, valid_dataset_x, valid_dataset_y)):
                
                inputs_tensor = self.torch_model.embedding.process_x_dataset(inputs)
                labels_tensor = self.torch_model.embedding.process_y_dataset(labels)
                
                inputs_tensor = torch.LongTensor(inputs_tensor).to(self.device)
                labels_tensor = torch.LongTensor(labels_tensor).to(self.device)
                
                outputs = self.torch_model(inputs_tensor)
                
                loss = self.loss_func(outputs.to(self.device),
                                      labels_tensor.to(self.device))
                
                # softmax
                outputs = F.softmax(outputs, dim=1)
                probability, pred_idx = torch.max(outputs, dim=1)
                pred_idx = pred_idx.cpu().numpy()
                
                # reverse_map
                pred_class = self.torch_model.embedding.reverse_numerize_label_sequences(pred_idx)
                pred_y_list.extend(pred_class)
                true_y_list.extend(labels)
                loss_list.append(loss)
        
        # stat
        accuracy = compute_accuracy(pred_y_list, true_y_list,
                                    print_detail=print_detail)
        weighted_f1, average_f1 = compute_f1_single_label(
            pred_y_list, true_y_list, print_detail=print_detail)
        mean_loss = sum(loss_list) / len(loss_list)
        
        return mean_loss, accuracy, weighted_f1, average_f1
        
    def predict(self,
                input_data,
                batch_size=32,
                #multi_label_threshold: float = 0.5,
                with_probability: bool = False):
            
        if self.torch_model is None:
            raise ValueError('`torch_model` should NOT be None.')
            
        with torch.no_grad():
            self.torch_model.training = False
            predicted = list()
            for i, inputs in enumerate(
                self._dispatch_dataset(batch_size, input_data)):
                
                inputs_tensor = self.torch_model.embedding.process_x_dataset(inputs)
                inputs_tensor = torch.LongTensor(inputs_tensor).to(self.device)
                
                outputs = self.torch_model(inputs_tensor)
                
                # softmax
                outputs = F.softmax(outputs, dim=1)
                probability, pred_idx = torch.max(outputs, dim=1)
                pred_idx = pred_idx.cpu().numpy()
                
                # reverse_map
                pred_class = self.torch_model.embedding.reverse_numerize_label_sequences(pred_idx)
                predicted.extend(pred_class)

        return predicted
        
        


