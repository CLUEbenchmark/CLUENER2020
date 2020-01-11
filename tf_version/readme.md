

  # 如何训练、提交测试
  
  ## 模型1 一键运行RoBERTa-wwm-large
  
  nohup bash run_classifier_roberta_wwm_large.sh &

  
  ## 模型2
      
      第一步， 生成tf_record
      修改 data_processor_seq.py 里面 函数的输入输出路径即可
      ```
      python data_processor_seq.py
      ```
      
      第二步， 训练ner模型
      修改 train_sequence_label.py 里面 config字典即可（如模型参数、文件路径等）
      ```
      python train_sequence_label.py
      ```
      
      第三步， 加载模型进行测试
      修改 predict_sequence_label.py 里面 model_path（保存模型的路径）, 以及预测文件路径即可
      ```
      python predict_sequence_label.py
      ```
  
  ### 评估
  修改 score.py 里面 pre ，gold文件即可（验证可用），测试阶段不提供哦
  ```
  python score.py
  ```
  
| 模型     | <a href='https://www.cluebenchmarks.com/ner.html'>线上效果f1</a> |
|:-------------:|:-----:|
| bert-base   |  75.68  |
| roberta-wwm-large-ext | 79.88 |

各个实体的 得分情况f
```
TODO
```
