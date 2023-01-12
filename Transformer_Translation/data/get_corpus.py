import json
from tqdm import tqdm


def write_corpus(file_name, suffix, lines, split_rate=0.8):
    with open(f"{file_name}.{suffix}", "w+", encoding="utf-8") as fch:
        if file == "train":
            lines = lines[:int(len(lines) * split_rate)]
            fch.writelines(lines)
        elif file == "dev":
            lines = lines[int(len(lines) * split_rate):]
            fch.writelines(lines)
        else:
            lines = lines[int(len(lines) * split_rate):]
            fch.writelines(lines)


if __name__ == "__main__":
    ch_lines, en_lines = [], []
    files = ['train', 'dev', 'test']
    ch_path, en_path = 'corpus.target', 'corpus.source'
    corpus = json.load(open('ch2en_align_result.json', 'r', encoding="utf-8"))
    for item in tqdm(corpus):
        # ch_lines.append(" ".join(list(item['chinese_content'].replace(" ",""))) + "\n")
        ch_lines.append(item['chinese_content'].replace(" ","") + "\n")
        en_lines.append(item['english_content'] + "\n")
    for file in files:
        write_corpus(file, suffix="target", lines=ch_lines, split_rate=0.8)
        write_corpus(file, suffix="source", lines=en_lines, split_rate=0.8)
    print(f"模型训练的数据集：{int(len(ch_lines) * 0.8)}个，验证或测试数据集：{len(ch_lines) - int(len(ch_lines) * 0.8)}个")
