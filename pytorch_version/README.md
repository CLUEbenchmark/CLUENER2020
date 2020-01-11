# CLUE_NER_PyTorch

CLUENER 细粒度命名实体识别

## 数据介绍

数据详细描述: https://www.cluebenchmarks.com/introduce.html

## 代码目录说明

```text
├── callback  # 自定义常用callback
|  └── lr_scheduler.py　　
|  └── ...
├── losses   #　自定义loss函数
|  └── focal_loss.py　
|  └── ...
├── CLUEdatasets   #　存放数据
|  └── cluener　　　
├── metrics　　　　　　　　　# metric计算
|  └── ner_metrics.py　　　
├── outputs              # 模型输出保存
|  └── cluener_output
├── prev_trained_model　# 预训练模型
|  └── albert_base
|  └── bert-wwm
|  └── ...
├── processors　　　　　# 数据处理
|  └── ner_seq.py
|  └── ...
├── tools　　　　　　　　#　通用脚本
|  └── common.py
|  └── download_clue_data.py
|  └── ...
├── models　　　# 主模型
|  └── transformers
|  └── bert_for_ner.py
|  └── ...
├── run_ner_span.py       # 主程序
├── run_ner_span.sh   #　任务运行脚本
```
## 依赖模块

* pytorch=1.1.0
* boto3=1.9
* regex
* sacremoses
* sentencepiece
* python3.7+

## 运行方式

### 1. 下载CLUE_NER数据集，运行以下命令：
```python
python tools/download_clue_data.py --data_dir=./CLUEdatasets --tasks=cluener
```
### 2. 预训练模型文件格式，比如:
```text
├── prev_trained_model　# 预训练模型
|  └── bert-base
|  | └── vocab.txt
|  | └── config.json
|  | └── pytorch_model.bin
```
### 3. 直接运行对应模式sh脚本，如：
```shell
sh run_ner_xxx.sh
```
### 4. 评估

当前默认使用最后一个checkpoint模型作为评估模型，你也可以指定--predict_checkpoints参数进行对应的checkpoint进行评估，比如：

```python
CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export GLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"

python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_predict \
  --predict_checkpoints=100 \
  --do_lower_case \
  --loss_type=ce \
　...
```
### 模型列表

model_type目前支持**bert**和**albert**

**注意**: bert ernie bert_wwm bert_wwwm_ext等模型只是权重不一样，而模型本身主体一样，因此参数model_type=bert其余同理。

### 输入编码方式

目前默认为BIOS编码方式，比如:

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
谈	O
笑	O
风	O
生	O 
```

## 结果

以下为模型在 **dev**上的测试结果:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |                                                              |
| ------------ | ------------------ | ------------------ | ------------------ | ------------------------------------------------------------ |
| BERT+Softmax | 0.7916     | 0.7962     | 0.7939    | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 |
| BERT+CRF     | 0.7877     | 0.8008 | 0.7942     | train_max_length=128 eval_max_length=512 epoch=5 lr=3e-5 batch_size=24 |
| BERT+Span    | 0.8132 | **0.8092** | **0.8112** | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 |
| BERT+Span+focal_loss    | 0.8121 | 0.8008 | 0.8064 | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 loss_type=focal |
| BERT+Span+label_smoothing   | **0.8235** | 0.7946 | 0.8088 | train_max_length=128 eval_max_length=512 epoch=4 lr=3e-5 batch_size=24 loss_type=lsr |

