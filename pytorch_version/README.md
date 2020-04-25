### 数据介绍

数据详细描述: https://www.cluebenchmarks.com/introduce.html

### 运行方式
1. 下载CLUE_NER数据集，运行以下命令：
```shell
python tools/download_clue_data.py --data_dir=./datasets --tasks=cluener
```
2. 预训练模型文件格式，比如:
```text
├── prev_trained_model　# 预训练模型
|  └── bert-base
|  | └── vocab.txt
|  | └── config.json
|  | └── pytorch_model.bin
```
3. 训练：

直接执行对应shell脚本，如：
```shell
sh scripts/run_ner_crf.sh
```
4. 预测

当前默认使用最后一个checkpoint模型作为预测模型，你也可以指定--predict_checkpoints参数进行对应的checkpoint进行预测，比如：
```python
CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export CLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"

python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_predict \
  --predict_checkpoints=100 \
  --do_lower_case \
　...
```
### 模型列表

model_type目前支持**bert**和**albert**

**注意:** bert ernie bert_wwm bert_wwwm_ext等模型只是权重不一样，而模型本身主体一样，因此参数model_type=bert其余同理。

### 结果

在dev上为F1分数为0.8076