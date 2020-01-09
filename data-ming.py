#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2020-01-09 18:51
"""
import re
import json


def prepare_label():
    text = """
    地址（address）: 544
    书名（book）: 258
    公司（company）: 479
    游戏（game）: 281
    政府（government）: 262
    电影（movie）: 307
    姓名（name）: 710
    组织机构（organization）: 515
    职位（position）: 573
    景点（scene）: 288
    """

    a = re.findall(r"（(.*?)）", text.strip())
    print(a)
    label2id = {"O": 0}
    index = 1
    for i in a:
        label2id["S_" + i] = index
        label2id["B_" + i] = index + 1
        label2id["M_" + i] = index + 2
        label2id["E_" + i] = index + 3
        index += 4

    open("label2id.json", "w").write(json.dumps(label2id, ensure_ascii=False, indent=2))


def prepare_len_count():
    len_count = {}

    for line in open("data/thuctc_train.json"):
        if line.strip():
            _ = json.loads(line.strip())
            len_ = len(_["text"])
            if len_count.get(len_):
                len_count[len_] += 1
            else:
                len_count[len_] = 1

    for line in open("data/thuctc_valid.json"):
        if line.strip():
            _ = json.loads(line.strip())
            len_ = len(_["text"])
            if len_count.get(len_):
                len_count[len_] += 1
            else:
                len_count[len_] = 1

    print("len_count", json.dumps(len_count, indent=2))
    open("len_count.json", "w").write(json.dumps(len_count, indent=2))


prepare_label()
