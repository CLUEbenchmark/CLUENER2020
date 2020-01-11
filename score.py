#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2020-01-09 22:53
"""
import json


def get_f1_score_label(pre_lines, gold_lines, label="organization"):
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


if __name__ == "__main__":
    f_score, avg = get_f1_score(pre_file="ner_predict_large.json", gold_file="data/thuctc_valid.json")

    print(f_score, avg)
