# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: measures.py
# time: 2020-06-12 11:27

import pdb
import logging
import numpy as np


def compute_accuracy(pred_labels, true_labels, print_detail=True):
    assert len(pred_labels) == len(true_labels)
    
    correct_num = 0
    total_num = len(pred_labels)
    for p_label, t_label in zip(pred_labels, true_labels):
        if p_label == t_label:
            correct_num += 1
    
    if print_detail:
        print('\n\tAccuracy: {:.2%}'.format(correct_num / total_num))
        
    return correct_num / total_num


def compute_f1_single_label(pred_labels, true_labels,
                            best_f1=None, print_detail=True):
    ''' 分类模型的 f1 值计算 '''
    assert len(pred_labels) == len(true_labels)
    true_stats = dict()
    pred_stats = dict()
    pred_true_stats = dict()
    
    for p_label, t_label in zip(pred_labels, true_labels):
        if t_label not in true_stats:
            true_stats[t_label] = 0
        true_stats[t_label] += 1
        
        if p_label not in pred_stats:
            pred_stats[p_label] = 0
        pred_stats[p_label] += 1
        
        if p_label == t_label:
            if p_label not in pred_true_stats:
                pred_true_stats[p_label] = 0
            pred_true_stats[p_label] += 1
            
    precisions = dict()
    recalls = dict()
    f1s = dict()
    for _class, num in true_stats.items():
        # compute precision
        precisions[_class] = 0.
        if _class in pred_stats:
            precisions[_class] = pred_true_stats.get(_class, 0.) / float(pred_stats.get(_class, 0.))
        
        # compute recall
        recalls[_class] = 0.
        recalls[_class] = pred_true_stats.get(_class, 0.) / float(true_stats.get(_class, 0.))
        
        # compute f1
        f1s[_class] = 0.
        if precisions[_class] + recalls[_class] != 0.:
            f1s[_class] = precisions[_class] * recalls[_class] * 2 / (precisions[_class] + recalls[_class])
    
    #pdb.set_trace()
    weighted_f1 = 0.
    average_f1 = 0.
    for k, v in f1s.items():
        average_f1 += v * (1 / len(f1s))
        weighted_f1 += v * (true_stats.get(k) / len(true_labels))

    if print_detail:
        print('\n\tAverage_F1: {:.2%}\n\tWeighted_F1: {:.2%}'.format(
            average_f1, weighted_f1))

    # ave_recall = sum(preds_stats.values())
    # av_r = ave_recall/total
    # print ('\tave_recall: %f' % av_r)
    if print_detail:
        print('\tClass\t\t\tF1\tPrecision\tRecall')
        for i in f1s:
            print('\t{:}\t\t{:.2%}\t{:.2%}[{:d}/{:d}]\t{:.2%}[{:d}/{:d}]'.format(
                i, f1s.get(i, 0), precisions.get(i, 0),
                pred_true_stats.get(i, 0), pred_stats.get(i, 0),
                recalls.get(i, 0), pred_true_stats.get(i, 0),
                true_stats.get(i, 0)))
    return weighted_f1, average_f1





