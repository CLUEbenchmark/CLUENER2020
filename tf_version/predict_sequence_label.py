#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-12-07 20:51
"""
import os
import re
import json
import tensorflow as tf
import tokenization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vocab_file = "./vocab.txt"
tokenizer_ = tokenization.FullTokenizer(vocab_file=vocab_file)
label2id = json.loads(open("./label2id.json").read())
id2label = [k for k, v in label2id.items()]


def process_one_example_p(tokenizer, text, max_seq_len=128):
    textlist = list(text)
    tokens = []
    # labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
        # labels = labels[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        # label_ids.append(label2id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids)
    return feature


def load_model(model_folder):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder, repr(e))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.reset_default_graph()
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We start a session and restore the graph weights
    sess_ = tf.Session()
    saver.restore(sess_, input_checkpoint)

    # opts = sess_.graph.get_operations()
    # for v in opts:
    #     print(v.name)
    return sess_


model_path = "./ner_bert_base/"
sess = load_model(model_path)
input_ids = sess.graph.get_tensor_by_name("input_ids:0")
input_mask = sess.graph.get_tensor_by_name("input_mask:0")  # is_training
segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")  # fc/dense/Relu  cnn_block/Reshape
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
p = sess.graph.get_tensor_by_name("loss/ReverseSequence_1:0")


def predict(text):
    data = [text]
    # 逐个分成 最大62长度的 text 进行 batch 预测
    features = []
    for i in data:
        feature = process_one_example_p(tokenizer_, i, max_seq_len=64)
        features.append(feature)
    feed = {input_ids: [feature[0] for feature in features],
            input_mask: [feature[1] for feature in features],
            segment_ids: [feature[2] for feature in features],
            keep_prob: 1.0
            }

    [probs] = sess.run([p], feed)
    result = []
    for index, prob in enumerate(probs):
        for v in prob[1:len(data[index]) + 1]:
            result.append(id2label[int(v)])
    print(result)
    labels = {}
    start = None
    index = 0
    for w, t in zip("".join(data), result):
        if re.search("^[BS]", t):
            if start is not None:
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label] = {te_: [[start, index - 1]]}
            start = index
            # print(start)
        if re.search("^O", t):
            if start is not None:
                # print(start)
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label] = {te_: [[start, index - 1]]}
            # else:
            #     print(start, labels)
            start = None
        index += 1
    if start is not None:
        # print(start)
        label = result[start][2:]
        if labels.get(label):
            te_ = text[start:index]
            # print(te_, labels)
            labels[label][te_] = [[start, index - 1]]
        else:
            te_ = text[start:index]
            # print(te_, labels)
            labels[label] = {te_: [[start, index - 1]]}
    # print(labels)
    return labels


def submit(path):
    data = []
    for line in open(path):
        if not line.strip():
            continue
        _ = json.loads(line.strip())
        res = predict(_["text"])
        data.append(json.dumps({"label": res}, ensure_ascii=False))
    open("ner_predict.json", "w").write("\n".join(data))


if __name__ == "__main__":
    text_ = "梅塔利斯在乌克兰联赛、杯赛及联盟杯中保持9场不败，状态相当出色；"
    res_ = predict(text_)
    print(res_)

    submit("data/thuctc_valid.json")
