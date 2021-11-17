  # CLUENER 细粒度命名实体识别 
  
  **更多细节请参考我们的 <a href='https://github.com/CLUEbenchmark/CLUENER2020/blob/master/CLUENER2020_paper.pdf'>技术报告</a>： https://arxiv.org/abs/2001.04351**
  ![./pics/header.png](https://github.com/CLUEbenchmark/CLUENER2020/blob/master/cluener.png)

  ## 数据类别：
    数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（government），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

  ## 标签类别定义 & 标注规则：
    地址（address）: **省**市**区**街**号，**路，**街道，**村等（如单独出现也标记）。地址是标记尽量完全的, 标记到最细。
    书名（book）: 小说，杂志，习题集，教科书，教辅，地图册，食谱，书店里能买到的一类书籍，包含电子书。
    公司（company）: **公司，**集团，**银行（央行，中国人民银行除外，二者属于政府机构）, 如：新东方，包含新华网/中国军网等。
    游戏（game）: 常见的游戏，注意有一些从小说，电视剧改编的游戏，要分析具体场景到底是不是游戏。
    政府（government）: 包括中央行政机关和地方行政机关两级。 中央行政机关有国务院、国务院组成部门（包括各部、委员会、中国人民银行和审计署）、国务院直属机构（如海关、税务、工商、环保总局等），军队等。
    电影（movie）: 电影，也包括拍的一些在电影院上映的纪录片，如果是根据书名改编成电影，要根据场景上下文着重区分下是电影名字还是书名。
    姓名（name）: 一般指人名，也包括小说里面的人物，宋江，武松，郭靖，小说里面的人物绰号：及时雨，花和尚，著名人物的别称，通过这个别称能对应到某个具体人物。
    组织机构（organization）: 篮球队，足球队，乐团，社团等，另外包含小说里面的帮派如：少林寺，丐帮，铁掌帮，武当，峨眉等。
    职位（position）: 古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等。
    景点（scene）: 常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等。
  
  ## 数据下载地址：
  <a href='https://www.cluebenchmarks.com/introduce.html'>数据下载</a>
    
  ## 数据分布：
    训练集：10748
    验证集集：1343

    按照不同标签类别统计，训练集数据分布如下（注：一条数据中出现的所有实体都进行标注，如果一条数据出现两个地址（address）实体，那么统计地址（address）类别数据的时候，算两条数据）：
    【训练集】标签数据分布如下：
    地址（address）:2829
    书名（book）:1131
    公司（company）:2897
    游戏（game）:2325
    政府（government）:1797
    电影（movie）:1109
    姓名（name）:3661
    组织机构（organization）:3075
    职位（position）:3052
    景点（scene）:1462

    【验证集】标签数据分布如下：
    地址（address）:364
    书名（book）:152
    公司（company）:366
    游戏（game）:287
    政府（government）:244
    电影（movie）:150
    姓名（name）:451
    组织机构（organization）:344
    职位（position）:425
    景点（scene）:199


  ## 数据字段解释：
    以train.json为例，数据分为两列：text & label，其中text列代表文本，label列代表文本中出现的所有包含在10个类别中的实体。
    例如：
      text: "北京勘察设计协会副会长兼秘书长周荫如"
      label: {"organization": {"北京勘察设计协会": [[0, 7]]}, "name": {"周荫如": [[15, 17]]}, "position": {"副会长": [[8, 10]], "秘书长": [[12, 14]]}}
      其中，organization，name，position代表实体类别，
      "organization": {"北京勘察设计协会": [[0, 7]]}：表示原text中，"北京勘察设计协会" 是类别为 "组织机构（organization）" 的实体, 并且start_index为0，end_index为7 （注：下标从0开始计数）
      "name": {"周荫如": [[15, 17]]}：表示原text中，"周荫如" 是类别为 "姓名（name）" 的实体, 并且start_index为15，end_index为17
      "position": {"副会长": [[8, 10]], "秘书长": [[12, 14]]}：表示原text中，"副会长" 是类别为 "职位（position）" 的实体, 并且start_index为8，end_index为10，同时，"秘书长" 也是类别为 "职位（position）" 的实体,
      并且start_index为12，end_index为14

## 数据来源：
    本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.

## 效果对比

  | 模型     | <a href='https://www.cluebenchmarks.com/ner.html'>线上效果f1</a> |
|:-------------:|:-----:|
| Bert-base   |  78.82  |
| RoBERTa-wwm-large-ext | 80.42 |
| Bi-Lstm + CRF | 70.00 |

各个实体的评测结果(F1 score)：

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
| Overall@Macro |   70.00 | 78.82  | **80.42** | 63.41  |

## 基线模型（一键运行）

  1.tf版本bert系列：<a href='https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version'>tf_version</a>
  (test, f1 80.42) 
  
  2.pytorch版本baseline：<a href='https://github.com/CLUEbenchmark/CLUENER2020/tree/master/pytorch_version'>pytorch_version</a>(79.63) 
 
  3.bilistm+crf的baseline: <a href="https://github.com/CLUEbenchmark/CLUENER2020/tree/master/bilstm_crf_pytorch">bilstm+crf</a>
  (test, f1 70.0) 

#### 技术交流与问题讨论QQ群: 836811304 Join us on QQ group


#### 引用我们 Cite Us

如果本目录中的内容对你的研究工作有所帮助，请在文献中引用下述报告：https://arxiv.org/abs/2001.04351
```
@article{xu2020cluener2020,
  title={CLUENER2020: Fine-grained Name Entity Recognition for Chinese},
  author={Xu, Liang and Dong, Qianqian and Yu, Cong and Tian, Yin and Liu, Weitang and Li, Lu and Zhang, Xuanwei},
  journal={arXiv preprint arXiv:2001.04351},
  year={2020}
 }
```
