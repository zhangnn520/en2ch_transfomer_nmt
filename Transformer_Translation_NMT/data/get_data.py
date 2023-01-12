#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import json
from utils.tokenize import run
from utils.data_helpers import read_json, write_json

base_path = os.path.dirname(__file__)
train_json_path = os.path.join(base_path, "train.json")
dev_json_path = os.path.join(base_path, "dev.json")
test_json_path = os.path.join(base_path, "test.json")
ch_input = os.path.join(base_path, "corpus.ch")
en_input = os.path.join(base_path, "corpus.en")


def generate_corpus():
    files = ['train', 'dev', 'test']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines = []
    en_lines = []

    for file in files:
        corpus = json.load(open(os.path.join(base_path, f"{file}.json"), 'r', encoding="utf-8"))
        for item in corpus:
            ch_lines.append(item[1] + '\n')
            en_lines.append(item[0] + '\n')

    with open(ch_path, "w", encoding="utf-8") as fch:
        fch.writelines(ch_lines)

    with open(en_path, "w", encoding="utf-8") as fen:
        fen.writelines(en_lines)

    # lines of Chinese: 252777
    print("lines of Chinese: ", len(ch_lines))
    # lines of English: 252777
    print("lines of English: ", len(en_lines))
    print("-------- Get Corpus ! --------")


if __name__ == "__main__":
    content_list = [[i['text'].split(" ||| ")[1], i['text'].split(" ||| ")[0].replace(" ", "")] for i in
                    read_json(os.path.join(base_path, "第1-3批经过校验后的数据.json"))]
    train_data = content_list[:int(len(content_list) * 0.9)]
    dev_data = content_list[int(len(content_list) * 0.9):]
    test_data = dev_data
    print(f"用于模型训练的数据数量为{len(train_data)}条,验证数据集{len(dev_data)}条")
    write_json(train_json_path, train_data)
    write_json(dev_json_path, dev_data)
    write_json(test_json_path, test_data)
    print("生成中英文corpus预料数据集，用于后续token处理")
    generate_corpus()
    print("对模型训练数据进行token处理，并进行bpe分词")
    run(ch_input, en_input)
