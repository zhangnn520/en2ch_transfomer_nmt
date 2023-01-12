import os
import time
import torch
from config.config import Config
from model.TranslationModel import TranslationModel
from utils.data_helpers import LoadEnglishChineseDataset, my_tokenizer, logger


class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr


def accuracy(logits, y_true, PAD_IDX):
    """
    :param logits:  [tgt_len,batch_size,tgt_vocab_size]
    :param y_true:  [tgt_len,batch_size]
    :param PAD_IDX:
    :return:
    """
    y_pred = logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [tgt_len,batch_size,tgt_vocab_size] 转成 [batch_size, tgt_len,tgt_vocab_size]
    y_true = y_true.transpose(0, 1).reshape(-1)
    # 将 [tgt_len,batch_size] 转成 [batch_size， tgt_len]
    acc = y_pred.eq(y_true)  # 计算预测值与正确值比较的情况
    mask = torch.logical_not(y_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    acc = acc.logical_and(mask)  # 去掉acc中mask的部分
    correct = acc.sum().item()
    total = mask.sum().item()
    return float(correct) / total, correct, total


def train_model(train_config):
    logger.info(f"Model train on device: {train_config.device}")
    data_loader = LoadEnglishChineseDataset(train_config.train_corpus_file_paths,
                                            batch_size=train_config.batch_size,
                                            tokenizer=my_tokenizer,
                                            min_freq=train_config.min_freq)

    train_iter, valid_iter, test_iter = \
        data_loader.load_train_val_test_data(train_config.train_corpus_file_paths,
                                             train_config.val_corpus_file_paths,
                                             train_config.test_corpus_file_paths)
    translation_model = TranslationModel(src_vocab_size=len(data_loader.ch_vocab),
                                         tgt_vocab_size=len(data_loader.en_vocab),
                                         d_model=train_config.d_model,
                                         n_head=train_config.num_head,
                                         num_encoder_layers=train_config.num_encoder_layers,
                                         num_decoder_layers=train_config.num_decoder_layers,
                                         dim_feedforward=train_config.dim_feedforward,
                                         dropout=train_config.dropout)

    translation_model = translation_model.to(train_config.device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_loader.PAD_IDX)

    optimizer = torch.optim.Adam(translation_model.parameters(), lr=0.,
                                 betas=(train_config.beta1, train_config.beta2), eps=train_config.epsilon)
    lr_scheduler = CustomSchedule(train_config.d_model, optimizer=optimizer, warmup_steps=train_config.warmup_steps)
    translation_model.train()
    num = 0
    for epoch in range(train_config.epochs):
        losses = 0
        total_acc = 0
        start_time = time.time()
        for idx, (src, tgt) in enumerate(train_iter):
            src = src.to(train_config.device)  # [src_len, batch_size]
            tgt = tgt.to(train_config.device)
            tgt_input = tgt[:-1, :]  # 解码部分的输入, [tgt_len,batch_size]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                data_loader.create_mask(src, tgt_input, train_config.device)
            logits = translation_model(
                src=src,  # Encoder的token序列输入，[src_len,batch_size]
                tgt=tgt_input,  # Decoder的token序列输入,[tgt_len,batch_size]
                src_mask=src_mask,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                tgt_mask=tgt_mask,
                # Decoder的注意力Mask输入，用于掩盖当前position之后的position [tgt_len,tgt_len]
                src_key_padding_mask=src_padding_mask,  # 用于mask掉Encoder的Token序列中的padding部分
                tgt_key_padding_mask=tgt_padding_mask,  # 用于mask掉Decoder的Token序列中的padding部分
                memory_key_padding_mask=src_padding_mask)  # 用于mask掉Encoder的Token序列中的padding部分
            # logits 输出shape为[tgt_len,batch_size,tgt_vocab_size]
            optimizer.zero_grad()
            tgt_out = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # [tgt_len*batch_size, tgt_vocab_size] with [tgt_len*batch_size, ]
            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            losses += loss.item()
            acc, _, _ = accuracy(logits, tgt_out, data_loader.PAD_IDX)
            total_acc += acc
            # msg = f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}],
            # Train loss :{loss.item():.3f}, Train acc: {acc:.5f}"
            # logger.info(msg)
        average_acc = total_acc / len(train_iter)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train_acc: {average_acc:.3f} " \
              f"Epoch time : {(end_time - start_time):.3f}s"
        logger.info(msg)

        if (epoch % train_config.eval_ratiocinate_step == 0) and train_config.do_eval:
            logger.info("Model eval on validation")
            acc = evaluate(train_config, valid_iter, translation_model, data_loader)

            if train_config.acc_threshold_value < acc:
                num += 1
                save_model_file_dir = os.path.join(train_config.model_save_dir, f"checkpoint_{num}_acc_{acc:.3f}")
                if not os.path.exists(save_model_file_dir):
                    os.makedirs(save_model_file_dir)
                save_model_path = os.path.join(save_model_file_dir, "pytorch_model.bin")
                # Good practice: save your training arguments together with the trained model
                torch.save(translation_model.state_dict(), save_model_path)
                logger.info(f"Acc: {acc:.3f} Greater the setting acc {config.acc_threshold_value},"
                            f"successfully save the model to {save_model_path}\n")
            logger.info(f"Accuracy on validation : {acc:.3f}")


def evaluate(eval_config, valid_iter, model, data_loader):
    model.eval()
    correct, totals = 0, 0
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(valid_iter):
            src = src.to(eval_config.device)
            tgt = tgt.to(eval_config.device)
            tgt_input = tgt[:-1, :]  # 解码部分的输入

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                data_loader.create_mask(src, tgt_input, device=eval_config.device)

            logits = model(src=src,  # Encoder的token序列输入，
                           tgt=tgt_input,  # Decoder的token序列输入
                           src_mask=src_mask,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                           tgt_mask=tgt_mask,  # Decoder的注意力Mask输入，用于掩盖当前position之后的position
                           src_key_padding_mask=src_padding_mask,  # 用于mask掉Encoder的Token序列中的padding部分
                           tgt_key_padding_mask=tgt_padding_mask,  # 用于mask掉Decoder的Token序列中的padding部分
                           memory_key_padding_mask=src_padding_mask)  # 用于mask掉Encoder的Token序列中的padding部分
            tgt_out = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
            _, c, t = accuracy(logits, tgt_out, data_loader.PAD_IDX)
            correct += c
            totals += t
    model.train()
    return float(correct) / totals


if __name__ == '__main__':
    config = Config()
    logger = logger(config)
    if config.do_train:
        train_model(config)
