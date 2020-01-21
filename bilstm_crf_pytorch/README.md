## CLUE_NER_PyTorch

CLUENER 细粒度命名实体识别

### 数据介绍

数据详细描述: https://www.cluebenchmarks.com/introduce.html

### 依赖模块

* pytorch=1.13.0
* python3.7+

### 运行方式

1. 下载CLUE_NER数据集，运行以下命令：

```python
python download_clue_data.py --data_dir=./dataset --tasks=cluener
```

2. 运行下列命令，进行模型训练：

   ```python
   python run_lstm_crf.py --do_train
   ```

3. 运行下列命令，进行模型预测

   ```python
   python run_lstm_crf.py --do_predict
   ```

   
