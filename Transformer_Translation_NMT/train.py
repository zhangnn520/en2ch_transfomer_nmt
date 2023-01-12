import torch
import warnings
from config.config import Config
from torch.utils.data import DataLoader
from model import make_model, LabelSmoothing
from utils.data_helpers import MTDataset, logger, train_evaluate, get_std_opt

warnings.filterwarnings("ignore")


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


def train_model(train_config):
    logger.info(f"Model train on device: {train_config.device}")
    train_dataset = MTDataset(train_config.train_corpus_file_paths)
    dev_dataset = MTDataset(train_config.val_corpus_file_paths)

    logger.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    logger.info("-------- Init transformer model! --------")
    # 初始化模型
    model = make_model(src_vocab=config.src_vocab_size,
                       tgt_vocab=config.tgt_vocab_size,
                       N=config.num_encoder_layers,
                       d_model=config.d_model,
                       d_ff=config.d_ff,
                       h=config.num_head,
                       dropout=config.dropout)

    logger.info("-------- Model parameter setting! --------")
    if not config.model_init:
        model.load_state_dict(torch.load(config.pretrain_model_path))
    model_par = torch.nn.DataParallel(model)
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=config.padding_idx, reduction='sum')

    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if config.use_smoothing:  # 降低我们对于标签的信任,一定程度上限制模型过拟合
        loss_fn = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        loss_fn.cuda()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.PAD, reduction='sum')

    train_evaluate(train_dataloader, dev_dataloader, model, model_par, loss_fn, optimizer, logger)


if __name__ == '__main__':
    config = Config()
    logger = logger(config)
    if config.do_train:
        train_model(config)
