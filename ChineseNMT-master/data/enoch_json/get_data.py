#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
from tools.tools import read_json, write_json

base_path = os.path.dirname(__file__)
train_json_path = os.path.join(base_path, "train.json")
dev_json_path = os.path.join(base_path, "dev.json")
test_json_path = os.path.join(base_path, "test.json")
content_list = [[i['text'].split(" ||| ")[1], i['text'].split(" ||| ")[0].replace(" ", "")] for i in
                read_json(os.path.join(base_path, "第1-3批经过校验后的数据.json"))]
train_data = content_list[:int(len(content_list) * 0.9)]
dev_data = content_list[int(len(content_list) * 0.9):]
test_data = dev_data
print(f"用于模型训练的数据数量为{len(train_data)}条,验证数据集{len(dev_data)}条")
write_json(train_json_path, train_data)
write_json(dev_json_path, dev_data)
write_json(test_json_path, test_data)
