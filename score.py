#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2020-01-09 22:53
"""
import json


def get_f1_score_label(pre_lines, gold_lines, pre_file="ner_predict.json", gold_file="data/thuctc_valid.json",
                       label="organization"):
    """
    打分函数
    """
    # pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    # gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    TP = 0
    FP = 0
    FN = 0
    for pre, gold in zip(pre_lines, gold_lines):
        pre = pre["label"].get(label, {}).keys()
        gold = gold["label"].get(label, {}).keys()
        for i in pre:
            if i in gold:
                TP += 1
            else:
                FP += 1
        for i in gold:
            if i not in pre:
                FN += 1
    print(TP, FP, FN)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f = 2 * p * r / (p + r)
    print(p, r, f)
    return f


def get_f1_score(pre_file="ner_predict.json", gold_file="data/thuctc_valid.json"):
    pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    f_score = {}
    labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    sum = 0
    for label in labels:
        f = get_f1_score_label(pre_lines, gold_lines, label=label)
        f_score[label] = f
        sum += f
    avg = sum / len(labels)
    return f_score, avg


f_score, avg = get_f1_score(pre_file="ner_predict_large.json", gold_file="data/thuctc_valid.json")

print(f_score, avg)

"""
bert base
{'address': 0.5180533751962323, 'book': 0.6690909090909091, 'company': 0.6646525679758307, 'game': 0.7094188376753506, 'government': 0.7352297592997812, 'movie': 0.5974025974025975, 'name': 0.7596513075965131, 'organization': 0.5411334552102377, 'position': 0.7240051347881901, 'scene': 0.553191489361702} 0.6471829433597345

roberta large wwm
{'address': 0.5081723625557206, 'book': 0.7333333333333334, 'company': 0.6906906906906908, 'game': 0.7338709677419354, 'government': 0.7428571428571429, 'movie': 0.6333333333333333, 'name': 0.7742749054224464, 'organization': 0.5494505494505494, 'position': 0.7039800995024875, 'scene': 0.5818181818181818} 0.6651781566705821
"""
