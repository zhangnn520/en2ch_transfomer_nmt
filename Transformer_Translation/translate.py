from config.config import Config
from model.TranslationModel import TranslationModel
from utils.data_helpers import LoadEnglishChineseDataset, my_tokenizer
import torch


def greedy_decode(model, SRC, max_len, start_symbol, Conf, data_loader):
    SRC = SRC.to(Conf.device)
    memory = model.encoder(src)  # 对输入的Token序列进行解码翻译
    ys = torch.ones(1, 1).fill_(start_symbol). \
        type(torch.long).to(Conf.device)  # 解码的第一个输入，起始符号
    for i in range(max_len - 1):
        memory = memory.to(Conf.device)
        out = model.decoder(ys, memory)  # [tgt_len,1,embed_dim]
        out = out.transpose(0, 1)  # [1,tgt_len, embed_dim]
        prob = model.classification(out[:, -1])  # 只对对预测的下一个词进行分类
        # out[:,1] shape : [1,embed_dim],  prob shape:  [1,tgt_vocab_size]
        _, next_word = torch.max(prob, dim=1)  # 选择概率最大者
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(SRC.data).fill_(next_word)], dim=0)
        # 将当前时刻解码的预测输出结果，同之前所有的结果堆叠作为输入再去预测下一个词。
        if next_word == data_loader.EOS_IDX:  # 如果当前时刻的预测输出为结束标志，则跳出循环结束预测。
            break
    return ys


def translate(model, SRC, data_loader, COF):
    src_vocab = data_loader.ch_vocab
    tgt_vocab = data_loader.en_vocab
    src_tokenizer = data_loader.tokenizer['source']
    model.eval()
    tokens = [src_vocab.stoi[tok] for tok in src_tokenizer(SRC)]  # 构造一个样本
    num_tokens = len(tokens)
    SRC = (torch.LongTensor(tokens).reshape(num_tokens, 1))  # 将src_len 作为第一个维度
    with torch.no_grad():
        tgt_tokens = greedy_decode(
            model, SRC, max_len=num_tokens + 5,
            start_symbol=data_loader.BOS_IDX,
            Conf=COF,
            data_loader=data_loader).flatten()  # 解码的预测结果
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")


def translate_german_to_english(SRC, conf):
    data_loader = LoadEnglishChineseDataset(conf.train_corpus_file_paths, batch_size=conf.batch_size,
                                            tokenizer=my_tokenizer, min_freq=conf.min_freq)
    translation_model = TranslationModel(src_vocab_size=len(data_loader.ch_vocab),
                                         tgt_vocab_size=len(data_loader.en_vocab),
                                         d_model=conf.d_model,
                                         n_head=conf.num_head,
                                         num_encoder_layers=conf.num_encoder_layers,
                                         num_decoder_layers=conf.num_decoder_layers,
                                         dim_feedforward=conf.dim_feedforward,
                                         dropout=conf.dropout)
    translation_model = translation_model.to(conf.device)
    loaded_paras = torch.load(conf.predict_model_dir + '/model.pkl')
    translation_model.load_state_dict(loaded_paras)
    result = list()
    for src_content in SRC:
        res = translate(translation_model, src_content, data_loader, conf)
        result.append(res)
    return result


if __name__ == '__main__':
    srcs = ["disconnect the radiator outlet hose from the supermanifold note 1x spring clip",
            "put the vehicle on a 2post lift but do not raise it at this time"]
    tgts = ["断开 散热器出口软管 与 超级歧管 的 连接 注 1 个 弹簧夹 ",
            "将 车辆 停放在 双柱举升机 上 ， 但 此时 不要 升高 车辆 "]
    config = Config()
    results = translate_german_to_english(srcs, config)
    for src, tgt, r in zip(srcs, tgts, results):
        print(f"英语：{src}")
        print(f"翻译：{r}")
        print(f"汉语：{tgt}")
        print("\n")
