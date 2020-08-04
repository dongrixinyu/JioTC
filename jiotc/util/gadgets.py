# -*- coding=utf-8 -*-

import os
import random


def dataset_spliter(dataset_x, dataset_y, 
                    ratio=[0.8, 0.05, 0.15],
                    shuffle=True):
    """ 将数据集按照训练、验证、测试进行划分 """
    dataset = [[sample_x, sample_y] for sample_x, sample_y
               in zip(dataset_x, dataset_y)]
    
    random.shuffle(dataset)
    
    # 统计各个类别的数据数量及占比
    # class_list = list(set(dataset_y))
    # dataset_y = 
    
    
    tmp_ds = list()
    current = 0
    for s in ratio:
        num = int(len(dataset) * s)
        tmp_ds.append(dataset[current: current + num])
        current += num

    train_x = [item[0] for item in tmp_ds[0]]
    train_y = [item[1] for item in tmp_ds[0]]
    valid_x = [item[0] for item in tmp_ds[1]]
    valid_y = [item[1] for item in tmp_ds[1]]
    test_x = [item[0] for item in tmp_ds[2]]
    test_y = [item[1] for item in tmp_ds[2]]
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y


#def load_model(model_dir):
    
    
    
    






















