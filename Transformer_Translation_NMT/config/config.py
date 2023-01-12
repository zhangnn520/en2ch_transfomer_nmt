import os
import torch


class Config:
    """
    模型配置类,其实我并不喜欢这样配置参数，先用着吧
    """

    def __init__(self):
        # 数据集设置相关配置
        self.dropout = 0.1
        self.padding_idx = 0
        self.bos_idx = 2
        self.eos_idx = 3
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.train_corpus_file_paths = os.path.join(self.dataset_dir, 'train.json')  # 训练时解码器的输入
        self.val_corpus_file_paths = os.path.join(self.dataset_dir, 'dev.json')  # 验证时解码器的输入
        self.test_corpus_file_paths = os.path.join(self.dataset_dir, 'test.json')
        self.min_freq = 1  # 在构建词表的过程中滤掉词（字）频小于min_freq的词（字）
        # 模型训练、预测和验证
        self.use_noamopt = True
        self.do_predict = True
        self.lr = 1e-5
        self.do_eval = True
        self.do_train = True
        self.use_smoothing = True
        # 日志打印
        self.only_file = False
        # 模型相关配置
        self.model_init = False  # 是否需要从头进行训练模型，反之加载训练好的模型
        self.src_vocab_size = 5000
        self.tgt_vocab_size = 5000
        self.warmup_steps = 4000
        self.batch_size = 256
        self.d_model = 512
        self.num_head = 8
        # greed decode的最大句子长度
        self.max_len = 60
        # beam size for bleu
        self.beam_size = 3
        self.d_ff = 2048
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 512
        self.dropout = 0.2
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.epoch_num = 500
        self.eval_ratiocinate_step = 10
        self.best_bleu_score = 98
        self.logs_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_dir = os.path.join(self.project_dir, 'out')
        self.predict_model_dir = os.path.join(self.project_dir, 'out')
        self.pretrain_model_path = os.path.join(self.project_dir, 'pre_train_model',
                                                "epoch_260_dev_loss_0.14585678279399872_bleu_96.6197762466269.pth")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        # 制定cuda 设备id
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gpu_id = '0'
        self.device_id = [0]
