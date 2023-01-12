# coding=utf-8
import os
import re
import json
import xlrd
import jieba
import paddle
from functools import reduce
from tqdm import tqdm
from nltk.tokenize import word_tokenize

base_dir = os.getcwd()


# 用户自定义词典

def word_cut(content, lan_type="ch"):
    if lan_type == "ch":
        content = content.lower().replace("lh", "左").replace("rh", "右")
        paddle.enable_static()
        seg_list = jieba.cut(content, use_paddle=True)  # 使用paddle模式进行分词
    else:
        content = content.lower().replace("lh", "LH").replace("rh", "RH")
        seg_list = word_tokenize(content)
    return ' '.join(seg_list).replace(" / ", "/")


def read_text(path):
    with open(path, "r", encoding="utf-8") as reader:
        return [i.replace("\n","") for i in reader.readlines()]


def read_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.loads(reader.read())


def get_content(object_list, content_name="CONTENT"):
    return [content[f'{content_name}'] for content in object_list]


def write_ann_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            writer.write(content + "\n")


def rewrite_ann_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            writer.write(content)


def write_text(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for content in content_line:
            if content != content_line[-1]:
                writer.write(content.replace("  ", " ") + "\n")
            else:
                writer.write(content)


def write_text_one(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for num, content in enumerate(content_line):
            if content != content_line[-1]:
                writer.write(str(num) + "\t" + content.replace("  ", " ") + "\n")
            else:
                writer.write(str(num) + "\t" + content)


def write_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        writer.writelines(json.dumps(content_line, ensure_ascii=False, indent=4))


def write_line_json(write_path, content_line):
    with open(write_path, "w+", encoding="utf-8") as writer:
        for line in content_line:
            writer.write(json.dumps({line: content_line[line]}, ensure_ascii=False))
            writer.write("\n")


def get_maintain_clear_data(english_maintain, chinese_maintain):
    english_clear_data_list, chinses_clear_data_list = list(), list()
    for en in tqdm(english_maintain):
        for ch in chinese_maintain:
            if en['STATISTICS'] == ch['STATISTICS'] and en['HTML'].split("/")[-1] == ch['HTML'].split("/")[-1]:
                english_clear_data_list.append(en)
                chinses_clear_data_list.append(ch)
    try:
        assert len(english_clear_data_list) == len(chinses_clear_data_list)
    except Exception as e:
        print(e)
    return english_clear_data_list, chinses_clear_data_list


def fast_align_data(en_list, ch_list):
    en2ch_list = list()
    en2ch_dict = dict()
    assert len(en_list) == len(ch_list)
    for num, ch_content in enumerate(ch_list):
        en_content = en_list[num]
        new_content = str(en_content).replace("\n", "") + ' ||| ' + ch_content.replace("\n", "")
        en2ch_dict[ch_content.replace("\n", "").replace(" ", "")] = str(en_content).replace("\n", "")
        en2ch_list.append(new_content)
    return en2ch_list, en2ch_dict


def get_fast_align_data(en2chcontent, en2ch_align_content, word_align_result_path):
    """
    :param en2chcontent: 对齐原始预料
    :param en2ch_align_content: 对齐后产生的映射字典
    :param word_align_result_path: 写入对齐映射的文件目录
    :return: null
    """
    result_list = []
    for num, content in enumerate(en2chcontent):
        ens, chs = content.split("|||")
        id2en = {str(num): en for num, en in enumerate(ens.split(" "))}
        id2ch = {str(num): ch for num, ch in enumerate(chs.split(" "))}
        content_align = en2ch_align_content[num]
        en_word_mapping = [id2en[i.split("-")[0]] for i in content_align.replace("\n", "").split(" ")]
        ch_word_mapping = [id2ch[i.split("-")[1]] for i in content_align.replace("\n", "").split(" ")]
        result = dict(zip(en_word_mapping, ch_word_mapping))
        result_list.append(result)
    write_json(word_align_result_path, result_list)


def sub_word(string):
    pattern_one = u"\\(.*?\\)|\\（.*?）|\\[.*?]|-|“|”|``|''"
    result = re.sub(pattern_one, "", string)
    pattern_two = u" +"
    result = re.sub(pattern_two, " ", result)

    return result


def delete_duplicate_elements(list_data):
    return reduce(lambda x, y: x if y in x else x + [y], [[], ] + list_data)


def get_data_from_xlsx(path):
    chinese_sentence_list = []
    enlish_sentence_list = []
    # 打开excel
    wb = xlrd.open_workbook(path)
    # 按工作簿定位工作表
    sh = wb.sheet_by_name('维修手册详细步骤')
    # print(sh.nrows)  # 有效数据行数
    # print(sh.ncols)  # 有效数据列数
    # print(sh.cell(0, 0).value)  # 输出第一行第一列的值
    # print(sh.row_values(0))  # 输出第一行的所有值
    for i in range(sh.nrows):
        # 将数据和标题组合成字典
        xlsx_dict = dict(zip(sh.row_values(0), sh.row_values(i)))
        # 遍历excel，打印所有数据
        chinese_maintenance = xlsx_dict['维修内容（中文）']
        chinese_sentence_list.append(chinese_maintenance)
        english_maintenance = xlsx_dict['维修内容（英文）']
        enlish_sentence_list.append(english_maintenance)

    return chinese_sentence_list[1:], enlish_sentence_list[1:]


def split_data(dir_path, chinese_word_list):
    for num in range(int(len(chinese_word_list) / 100)):
        if (num + 1) != int(len(chinese_word_list) / 100):
            temp_list_one = chinese_word_list[num * 100:(num + 1) * 100]
            file_path = os.path.join(dir_path, f"长度大于20--2498个-第{num}批.txt")
            write_text_one(file_path, temp_list_one)
        else:
            temp_list = chinese_word_list[num * 100:]
            file_path = os.path.join(dir_path, f"度大于20--2498个-第{num}批.txt")
            write_text_one(file_path, temp_list)
