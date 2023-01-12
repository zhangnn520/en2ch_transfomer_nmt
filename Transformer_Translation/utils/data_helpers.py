import os
import time
import logging
from collections import Counter
from torchtext.vocab import Vocab
import torch
from config.config import Config
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

config = Config()


def my_tokenizer():
    tokenizer = dict()
    if config.english2chinese_translation:
        tokenizer['source'] = get_tokenizer('spacy', language='en_core_web_sm')  # 英语
        tokenizer['target'] = get_tokenizer('spacy', language='zh_core_web_sm')  # 汉语
    else:
        tokenizer['source'] = get_tokenizer('spacy', language='zh_core_web_sm')  # 汉语
        tokenizer['target'] = get_tokenizer('spacy', language='en_core_web_sm')  # 英语
    return tokenizer


def build_vocab(tokenizer, filepath, min_freq=1, specials=None):
    """
    vocab = Vocab(counter, specials=specials)

    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    # ['<unk>', '<pad>', '<bos>', '<eos>', '.', 'a', 'are', 'A', 'Two', 'in', 'men',...]
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；

    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    # {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3, '.': 4, 'a': 5, 'are': 6,...}
    print(vocab.stoi['are'])  # 通过单词返回得到词表中对应的索引
    """
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    counter = Counter()
    with open(filepath, encoding='utf8') as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=specials, min_freq=min_freq)


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class LoadEnglishChineseDataset:
    def __init__(self, train_file_paths=None, tokenizer=None, batch_size=2, min_freq=1):
        # 根据训练预料建立英语和汉语各自的字典
        self.tokenizer = tokenizer()
        self.ch_vocab = build_vocab(self.tokenizer['source'], filepath=train_file_paths[0], min_freq=min_freq)
        self.en_vocab = build_vocab(self.tokenizer['target'], filepath=train_file_paths[1], min_freq=min_freq)
        self.specials = ['<unk>', '<pad>', '<bos>', '<eos>']  # 特殊字符用于处理文本中特殊符号token
        self.PAD_IDX = self.ch_vocab['<pad>']
        self.BOS_IDX = self.ch_vocab['<bos>']
        self.EOS_IDX = self.ch_vocab['<eos>']
        self.batch_size = batch_size

    def data_process(self, filepaths):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        :param filepaths:
        :return:
        """
        data = []
        raw_ch_iter = iter(open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(open(filepaths[1], encoding="utf8"))
        path_list = [i.split('\\')[-1] for i in filepaths]
        logging.info(f"正在将数据集 {path_list} 转换成 Token ID ")
        for (raw_ch, raw_en) in tqdm(zip(raw_ch_iter, raw_en_iter), ncols=80):
            ch_tensor_ = torch.tensor([self.ch_vocab[token] for token in self.tokenizer['source'](raw_ch.rstrip("\n"))],
                                      dtype=torch.long)
            en_tensor_ = torch.tensor([self.en_vocab[token] for token in self.tokenizer['target'](raw_en.rstrip("\n"))],
                                      dtype=torch.long)
            data.append((ch_tensor_, en_tensor_))

        return data

    def load_train_val_test_data(self, train_file_paths, val_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths)
        val_data = self.data_process(val_file_paths)
        test_data = self.data_process(test_file_paths)

        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)
        return train_iter, valid_iter, test_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        ch_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            ch_batch.append(de_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 token
            en = torch.cat([torch.tensor([self.BOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0)
            en_batch.append(en)
        # 以最长的序列为标准进行填充
        ch_batch = pad_sequence(ch_batch, padding_value=self.PAD_IDX)  # [ch_len,batch_size]
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)  # [en_len,batch_size]
        return ch_batch, en_batch

    def create_mask(self, src, tgt, device='cpu'):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)  # [tgt_len,tgt_len]
        # Decoder的注意力Mask输入，用于掩盖当前position之后的position，所以这里是一个对称矩阵

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
        # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的，所以这里全是0

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        # False表示not masked, True表示masked
        # 用于mask掉Encoder的Token序列中的padding部分,[batch_size, src_len]
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        # 用于mask掉Decoder的Token序列中的padding部分,batch_size, tgt_len
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def logger(arg):
    # 创建一个日志器
    log = logging.getLogger('logger')
    # 判断处理器是否存在，如果有处理器就不添加，如果不存在处理器就添加处理器
    if not log.handlers:
        # 设定日志器的日志级别（如果不设定，默认展示WARNING级别以上的日志）
        log.setLevel(logging.DEBUG)
        # 创建一个处理器， StreamHandler() 控制台实现日志输出
        sh = logging.StreamHandler()
        # 创建处理器，FileHandler() 将日志输出到文件保存
        fh = logging.FileHandler(
            os.path.join(arg.logs_dir, f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}_log.txt'),
            encoding='utf-8')
        # datefmt 表示输出日期的格式
        lf = logging.Formatter(fmt='%(asctime)s | %(filename)s：%(lineno)d line | %(levelname)s | %(message)s',
                               datefmt='%Y_%m_%d %H:%M:%S')

        # 控制台输出日志
        # 在日志器中加入处理器
        log.addHandler(sh)
        # 处理器中设置日志输出格式
        sh.setFormatter(lf)
        # 给处理器设置级别
        sh.setLevel(logging.INFO)

        # 文件保存日志
        log.addHandler(fh)
        fh.setFormatter(lf)
    return log
