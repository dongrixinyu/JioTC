# JioTC
A Text Classification Tool based on PyTorch





```python
import torch

from jiotc.embeddings import BareEmbedding
from jiotc.processor import Processor
from jiotc.models import BiLSTMAttentionModel
from jiotc.models import ModelOperator
from jiotc.util import dataset_spliter, compute_f1_single_label
from jiotc.losses import FocalLoss


dataset_x = [item.split(',', 1)[1] for item in content[1:]]
dataset_y = [item.split(',', 1)[0] for item in content[1:]]

train_x, train_y, valid_x, valid_y, test_x, test_y = dataset_spliter(
    dataset_x, dataset_y, ratio=[0.8, 0.05, 0.15])

# 整理超参数
sequence_length = 200
embedding_size = 100
batch_size = 64
hidden_size = 50
dropout_rate = 0.2
epoch = 10
learning_rate = 9e-3

# 建立预处理和词表示
processor = Processor(
    multi_label=False)

bare_embed = BareEmbedding(
    processor=processor,
    embedding_weight=embedding_weight,
    sequence_length=sequence_length)
bare_embed.analyze_corpus(dataset_x, dataset_y)

# 建立模型
# 指定模型的超参数，以及 device
model_hyper = BiLSTMAttentionModel.get_default_hyper_parameters()

model_hyper['layer_bi_lstm']['hidden_size'] = hidden_size
model_hyper['layer_dense']['activation'] = 'softmax'  #
model_hyper['layer_bi_lstm']['dropout'] = dropout_rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bilstm_model = BiLSTMAttentionModel(bare_embed,
    hyper_parameters=model_hyper).to(device)

# 指定训练参数
training_hyper = ModelOperator.get_training_default_hyper_parameters()
print(training_hyper)
training_hyper['epoch'] = epoch
training_hyper['batch_size'] = batch_size
training_hyper['learning_rate'] = learning_rate

model_operator = ModelOperator(
    torch_model=bilstm_model,
    hyper_parameters=training_hyper)

model_operator.compile_model(
    optimizer=torch.optim.Adam(bilstm_model.parameters(), lr=learning_rate),
    loss_func=FocalLoss(alpha=0.2))
    #loss_func=nn.CrossEntropyLoss())

model_operator.train(train_x, train_y, valid_x, valid_y)

model_operator.evaluate(train_x, train_y)
model_operator.evaluate(valid_x, valid_y)
model_operator.evaluate(test_x, test_y)
model_operator.save('model.ckpt')


# 加载模型的测试
model_oprt = ModelOperator()
model_oprt.load('model.ckpt')

label = model_oprt.predict(
    list('房间比较差，尤其是洗手间，房间隔音和餐饮服务都不好。'))
print('predict label: ', label)

```



 
