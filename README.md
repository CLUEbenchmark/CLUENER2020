  CLUENER 细粒度命名实体识别 

  # 数据类别：
    数据分为10个标签类别，分别为: 地址（address），书名（book），公司（company），游戏（game），政府（government），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

  # 标签类别定义 & 标注规则：
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
  
  # 数据下载地址：
  <a href='http://www.cluebenchmark.com/introduce.html'>数据下载</a>
    
  # 数据分布：
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


  # 数据字段解释：
    以train.xlsx为例，数据分为两列：text & label，其中text列代表文本，label列代表文本中出现的所有包含在10个类别中的实体。
    例如：
      text: "北京勘察设计协会副会长兼秘书长周荫如"
      label: {"organization": {"北京勘察设计协会": [[0, 7]]}, "name": {"周荫如": [[15, 17]]}, "position": {"副会长": [[8, 10]], "秘书长": [[12, 14]]}}
      其中，organization，name，position代表实体类别，
      "organization": {"北京勘察设计协会": [[0, 7]]}：表示原text中，"北京勘察设计协会" 是类别为 "组织机构（organization）" 的实体, 并且start_index为0，end_index为7 （注：下标从0开始计数）
      "name": {"周荫如": [[15, 17]]}：表示原text中，"周荫如" 是类别为 "姓名（name）" 的实体, 并且start_index为15，end_index为17
      "position": {"副会长": [[8, 10]], "秘书长": [[12, 14]]}：表示原text中，"副会长" 是类别为 "职位（position）" 的实体, 并且start_index为8，end_index为10，同时，"秘书长" 也是类别为 "职位（position）" 的实体,
      并且start_index为12，end_index为14


  # 数据来源：
    本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.


  # 如何训练提交测试
  
  ## 第一步， 生成tf_record
  修改 data_processor_seq.py 里面 函数的输入输出路径即可
  ```
  python data_processor_seq.py
  ```
  
  ## 第二步， 训练ner模型
  修改 train_sequence_label.py 里面 config字典即可（如模型参数、文件路径等）
  ```
  python train_sequence_label.py
  ```
  
  ## 第三步， 加载模型进行测试
  修改 predict_sequence_label.py 里面 model_path（保存模型的路径）, 以及预测文件路径即可
  ```
  python predict_sequence_label.py
  ```
  
  # 评估
  修改 score.py 里面 pre ，gold文件即可（验证可用），测试阶段不提供哦
  ```
  python score.py
  ```
  
| 模型     | 效果 |
|:-------------:|:-----:|
| bert-base   |  0.647  |
| roberta-wwm-large-ext | 0.665  |

各个实体的 得分情况f
```
bert base
{'address': 0.5180533751962323, 'book': 0.6690909090909091, 'company': 0.6646525679758307, 'game': 0.7094188376753506, 'government': 0.7352297592997812, 'movie': 0.5974025974025975, 'name': 0.7596513075965131, 'organization': 0.5411334552102377, 'position': 0.7240051347881901, 'scene': 0.553191489361702}

roberta large wwm
{'address': 0.5081723625557206, 'book': 0.7333333333333334, 'company': 0.6906906906906908, 'game': 0.7338709677419354, 'government': 0.7428571428571429, 'movie': 0.6333333333333333, 'name': 0.7742749054224464, 'organization': 0.5494505494505494, 'position': 0.7039800995024875, 'scene': 0.5818181818181818}
```

#### 技术交流与问题讨论QQ群: 836811304 Join us on QQ group


#### Cite Us
CLUENER, CLUE Organization, (2020), GitHub repository, https://github.com/CLUEbenchmark/CLUENER

