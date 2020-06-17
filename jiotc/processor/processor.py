# -*- coding=utf-8 -*-

# author: dongrixinyu
# contact: dongrixinyu.89@163.com
# blog: https://github.com/dongrixinyu/

# file: processor.py
# time: 2020-06-12 11:27

import pdb
import copy
import json
import collections
import logging
import operator
from typing import List, Optional, Union, Dict, Any

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class Processor(object):
    
    def __init__(self, multi_label=False, **kwargs):
        self.token2idx: Dict[str, int] = kwargs.get('token2idx', {})
        self.idx2token: Dict[int, str] = dict(
            [(v, k) for (k, v) in self.token2idx.items()])

        self.token2count: Dict = {}

        self.label2idx: Dict[str, int] = kwargs.get('label2idx', {})
        self.idx2label: Dict[int, str] = dict(
            [(v, k) for (k, v) in self.label2idx.items()])

        self.token_pad: str = kwargs.get('token_pad', '<PAD>')
        self.token_unk: str = kwargs.get('token_unk', '<UNK>')
        self.token_bos: str = kwargs.get('token_bos', '<BOS>')
        self.token_eos: str = kwargs.get('token_eos', '<EOS>')
        self.token_sap: str = kwargs.get('token_eos', '<SAP>')

        self.dataset_info: Dict[str, Any] = kwargs.get('dataset_info', {})

        self.add_bos_eos: bool = kwargs.get('add_bos_eos', False)

        self.sequence_length = kwargs.get('sequence_length', None)

        self.min_count = kwargs.get('min_count', 3)
        
        self.multi_label = multi_label
        if self.label2idx:
            self.multi_label_binarizer: MultiLabelBinarizer = MultiLabelBinarizer(
                classes=list(self.label2idx.keys()))
            self.multi_label_binarizer.fit([])
        else:
            self.multi_label_binarizer: MultiLabelBinarizer = None
        
    def info(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'multi_label': self.multi_label,
                'label2idx': self.label2idx,
                'token2idx': self.token2idx,
                'token_pad': self.token_pad,
                'token_unk': self.token_unk,
                'token_bos': self.token_bos,
                'token_eos': self.token_eos,
                'dataset_info': self.dataset_info,
                'add_bos_eos': self.add_bos_eos,
                'sequence_length': self.sequence_length
            },
            'module': self.__class__.__module__
        }
    
    def analyze_corpus(self,
                       corpus: Union[List[List[str]]],
                       labels: Union[List[List[str]], List[str]],
                       force: bool = False):
        rec_len = sorted([len(seq) for seq in corpus])[int(0.95 * len(corpus))]
        self.dataset_info['RECOMMEND_LEN'] = rec_len

        if len(self.token2idx) == 0 or force:
            self._build_token_dict(corpus, self.min_count)
        if len(self.label2idx) == 0 or force:
            self._build_label_dict(labels)
        # pdb.set_trace()
            
    def _build_token_dict(self, corpus: List[List[str]], min_count: int = 3):
        """
        Build token index dictionary using corpus

        Args:
            corpus: List of tokenized sentences, like ``[['I', 'love', 'tf'], ...]``
            min_count:
        """
        token2idx = {
            self.token_pad: 0,
            self.token_unk: 1,
            self.token_bos: 2,
            self.token_eos: 3,
            self.token_sap: 4
        }

        token2count = {}
        for sentence in corpus:
            for token in sentence:
                count = token2count.get(token, 0)
                token2count[token] = count + 1
        self.token2count = token2count

        # 按照词频降序排序
        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        token2count = collections.OrderedDict(sorted_token2count)

        for token, token_count in token2count.items():
            if token not in token2idx and token_count >= min_count:
                token2idx[token] = len(token2idx)

        self.token2idx = token2idx
        self.idx2token = dict([(value, key)
                               for key, value in self.token2idx.items()])
        logging.debug(f"build token2idx dict finished, contains {len(self.token2idx)} tokens.")
        self.dataset_info['token_count'] = len(self.token2idx)

    def _build_label_dict(self, labels: List[str]):
        if self.multi_label:
            label_set = set()
            for i in labels:
                label_set = label_set.union(list(i))
        else:
            label_set = set(labels)
            
        self.label2idx = dict()
        for idx, label in enumerate(sorted(label_set)):
            self.label2idx[label] = len(self.label2idx)

        self.idx2label = dict([(value, key) for key, value in self.label2idx.items()])
        self.dataset_info['label_count'] = len(self.label2idx)
        self.multi_label_binarizer = MultiLabelBinarizer(
            classes=list(self.label2idx.keys()))

    def process_x_dataset(self,
                          data: List[List[str]],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if max_len is None:
            max_len = self.sequence_length
        if subset is not None:
            target = utils.get_list_subset(data, subset)
        else:
            target = data
        numerized_samples = self.numerize_token_sequences(target)
        #pdb.set_trace()
        return self.pad_sequences(
            numerized_samples, max_len,
            pad_token_idx=self.token2idx['<PAD>'], truncating=True)

    def pad_sequences(self, numerized_samples, max_len: int, 
                      pad_token_idx: int = 0,
                      truncating: bool = True):
        padded_numerized_samples = list()
        for sample in numerized_samples:
            if len(sample) < max_len:
                sample.extend([pad_token_idx for i in range(max_len - len(sample))])
                padded_numerized_samples.append(sample)
            elif len(sample) == max_len:
                padded_numerized_samples.append(sample)
            elif len(sample) > max_len:
                if truncating:
                    padded_numerized_samples.append(sample[: max_len])
                else:
                    padded_numerized_samples.append(sample)
                    
        return padded_numerized_samples
    
    def process_y_dataset(self,
                          data: List[str],
                          max_len: Optional[int] = None,
                          subset: Optional[List[int]] = None) -> np.ndarray:
        if subset is not None:
            target = utils.get_list_subset(data, subset)
        else:
            target = data
        if self.multi_label:
            return self.multi_label_binarizer.fit_transform(target)
        else:
            numerized_samples = self.numerize_label_sequences(target)
            return numerized_samples
            #pdb.set_trace()
            return self.to_categorical_onehot(
                numerized_samples, len(self.label2idx))

    def to_categorical_onehot(self, numerized_samples: List[int],
                              category_num: int):
        onehot_numerized_samples = list()
        empty_onehot = [0 for i in range(category_num)]
        for sample in numerized_samples:
            cur_onehot = copy.deepcopy(empty_onehot)
            cur_onehot[sample] = 1
            onehot_numerized_samples.append(cur_onehot)
            
        return onehot_numerized_samples
        
    def numerize_token_sequences(self,
                                 sequences: List[List[str]]):
        result = []
        for seq in sequences:
            if self.add_bos_eos:
                seq = [self.token_bos] + seq + [self.token_eos]
            unk_index = self.token2idx[self.token_unk]
            result.append([self.token2idx.get(token, unk_index) for token in seq])
        return result

    def numerize_label_sequences(self, sequences: List[str]) -> List[int]:
        """
        Convert label sequence to label-index sequence
        ``['O', 'O', 'B-ORG'] -> [0, 0, 2]``

        Args:
            sequences: label sequence, list of str

        Returns:
            label-index sequence, list of int
        """
        return [self.label2idx[label] for label in sequences]

    def reverse_numerize_label_sequences(self, sequences):
        if self.multi_label:
            return self.multi_label_binarizer.inverse_transform(sequences)
        else:
            return [self.idx2label[label] for label in sequences]

    def __repr__(self):
        return f"<{self.__class__}>"

    def __str__(self):
        return self.__repr__()

        
        
        
        
        
        
        
    def _read_pretrained_embedding_data(self, pre_trained_embedding_file,
                                        embedding_size):
        ''' 读取预训练的词向量文件 '''
        # 获取词向量，或字向量文件的绝对路径
        try:
            if pre_trained_embedding_file is None:
                pass  # 不加载词向量
            elif os.path.isabs(pre_trained_embedding_file):
                pass  # 若是绝对路径则直接跳过
            else:
                # 从该包的下载器中寻找模型
                pre_trained_embedding_file = os.path.join(
                    self.default_models_dir,
                    pre_trained_embedding_file,
                    pre_trained_embedding_file + '.wv')
        except Exception as e:
            logging.error(e)
            raise ValueError('Please download embedding file via ' \
                             '`bbd_tools.download()`')
            
        # 读取词向量或字向量的数据出来
        embedding_vector_data = dict()
        content = bbd.read_file_by_line(pre_trained_embedding_file)
        for idx, line in enumerate(content):
            if idx == 0:
                parts = line.strip().split(' ')
                logging.info('预训练词向量总个数：{}，词向量维度：{}'.format(
                    int(parts[0]), int(parts[1])))
                if int(parts[1]) != embedding_size:
                    raise ValueError('the designated embedding size is not' \
                                     'equal to {}'.format(int(parts[1])))
            else:
                parts = line.strip().split(' ')
                token = parts[0]
                embedding_vector = [float(i) for i in parts[1:]]
                embedding_vector_data.update({token: embedding_vector})
        return embedding_vector_data
        
    def _compute_text_2_token_idx(self, labeled_data, token_type,
                                  data_dir, word_seg_tool, 
                                  embed_vector_data):
        ''' 
        计算数据集的  token 个数 和 label 个数，并做映射，成 idx
        '''
        random.shuffle(labeled_data)  # 打散数据，为后续操作准备
        
        token_sample_list = list()
        token_list = list()
        label_list = list()
        for item in labeled_data:
            text = item['text']
            label_list.append(item['label'])
            if token_type == 'word':
                token_sample = bbd.word_segmenter(text, method=word_seg_tool)
                token_list.extend(token_sample)
                token_sample_list.append(token_sample)
            elif token_type == 'char':
                token_sample = list(text)
                token_list.extend(token_sample)
                token_sample_list.append(token_sample)
            else:
                raise ValueError('token_type must be `word` or `char`.')
                
        # 处理 token 映射表
        token_set = set(sorted(token_list))
        
        # 增加额外 token
        token_dict = {'<PAD>': 0, '<UNK>': 1}
        token_reversed_dict = {0: '<PAD>', 1: '<UNK>'}
        
        if len(token_set) < self.vocab_size:
            # 即训练数据单一或太少，词汇量不足要求的数量，应当从词向量中选择部分数据填充
            # token_type == char 时不会有该问题
            for idx, token in enumerate(list(token_set)):
                token_dict.update({token: len(token_dict)})
                token_reversed_dict.update(
                    {len(token_reversed_dict): token})

            # 补充 token
            if token_type == 'word':
                for t, v in embed_vector_data.items():
                    if t not in token_dict:
                        token_dict.update({t: len(token_dict)})
                        token_reversed_dict.update(
                            {len(token_reversed_dict): t})
                    
                    if len(token_dict.items()) == self.vocab_size:
                        break
        else:
            # 训练数据较多，需要对 token 数量做截断
            token_count = collections.Counter(
                token_list).most_common(self.vocab_size - 2)
            for idx, tup in enumerate(token_count):
                token_dict.update({tup[0]: len(token_dict)})
                token_reversed_dict.update(
                    {len(token_reversed_dict): tup[0]})
            
        # 打印各个类别样本数量
        label_res = collections.Counter(label_list).most_common()
        logging.info('number of all samples: {}.'.format(len(label_list)))
        for item in label_res:
            logging.info('`{0}`, num: {1}, ratio: {2:.2%}.'.format(
                item[0], item[1], item[1] / len(label_list)))
            
        # 制作 label 映射表
        label_set = list(set(label_list))
        label_dict = dict([(label, idx) for idx, label in enumerate(label_set)])
        label_reversed_dict = dict(
            [(idx, label) for idx, label in enumerate(label_set)])
        
        with open(os.path.join(data_dir, 'label_map.json'), 
                  'w', encoding='utf-8') as fw:
            json.dump(label_dict, fw, ensure_ascii=False)
            
        with open(os.path.join(data_dir, 'token_map.json'),
                  'w', encoding='utf-8') as fw:
            json.dump(token_dict, fw, ensure_ascii=False)
            
        return token_sample_list, label_list, token_dict, \
            token_reversed_dict, label_dict, label_reversed_dict
        

        
    def make_embedding_for_model(self, embedding_vector_data, 
                                 token_dict, embedding_size):
        ''' 制作可供模型装载的预训练词向量表征，若不在词向量里，就随机初始化 '''
        embedding_res = list()
        for token, idx in token_dict.items():
            if token in embedding_vector_data:
                embedding_res.append(embedding_vector_data[token])
            else:
                embedding_res.append(
                    list(np.random.random(embedding_size) - 0.5))
        return embedding_res
        

