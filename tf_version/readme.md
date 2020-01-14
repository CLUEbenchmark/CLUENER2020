

  # 如何训练、提交测试
  
  环境：Python 3 & Tensorflow 1.x，如1.14; 
  
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
  以F1-Score为评测指标，修改 score.py 里面 pre ，gold文件即可（验证可用），测试阶段不提供哦
  ```
  python score.py
  ```
  
| 模型     | <a href='https://www.cluebenchmarks.com/ner.html'>线上效果f1</a> |
|:-------------:|:-----:|
| bilstm+crf  |  70.00  |
| bert-base   |  78.82  |
| roberta-wwm-large-ext | **80.42** |
|Human Performance|63.41|

各个实体的评测结果：


| 实体     | bilstm+crf | bert-base | roberta-wwm-large-ext | Human Performance |
|:-------------:|:-----:|:-----:|:-----:|:-----:|
| Person Name   | 74.04 | 88.75 | **89.09** | 74.49 |
| Organization  | 75.96 | 79.43 | **82.34** | 65.41 |
| Position      | 70.16 | 78.89 | **79.62** | 55.38 |
| Company       | 72.27 | 81.42 | **83.02** | 49.32 |
| Address       | 45.50 | 60.89 | **62.63** | 43.04 |
| Game          | 85.27 | 86.42 | **86.80** | 80.39 |
| Government    | 77.25 | 87.03 | **88.17** | 79.27 |
| Scene         | 52.42 | 65.10 | **70.49** | 51.85 |
| Book          | 67.20 | 73.68 | **74.60** | 71.70 |
| Movie         | 78.97 | 85.82 | **87.46** | 63.21 |

更具体的评测结果，请参考我们的技术报告：https://arxiv.org/abs/2001.04351
