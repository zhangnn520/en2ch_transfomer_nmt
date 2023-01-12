import os
import torch


class Config:
    """
    基于Transformer架构的类Translation模型配置类
    """

    def __init__(self):
        # 设置英译汉或者汉译英，如果需要其他语言需要设置data_helpers.my_tokenizer()
        self.english2chinese_translation = False
        # 数据集设置相关配置
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.train_corpus_file_paths = [os.path.join(self.dataset_dir, 'train.source'),  # 训练时编码器的输入
                                        os.path.join(self.dataset_dir, 'train.target')]  # 训练时解码器的输入
        self.val_corpus_file_paths = [os.path.join(self.dataset_dir, 'dev.source'),  # 验证时编码器的输入
                                      os.path.join(self.dataset_dir, 'dev.target')]  # 验证时解码器的输入
        self.test_corpus_file_paths = [os.path.join(self.dataset_dir, 'test.source'),
                                       os.path.join(self.dataset_dir, 'test.target')]
        self.min_freq = 1  # 在构建词表的过程中滤掉词（字）频小于min_freq的词（字）
        # 模型训练、预测和验证
        self.do_train = True
        self.do_eval = True
        self.do_predict = True
        # 日志打印
        self.only_file = False
        # 模型相关配置
        self.warmup_steps = 4000
        self.batch_size = 256
        self.d_model = 512
        self.num_head = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 512
        self.dropout = 0.2
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 500
        self.eval_ratiocinate_step = 10
        self.acc_threshold_value = 0.80
        self.logs_dir = os.path.join(self.project_dir, 'logs')
        self.model_save_dir = os.path.join(self.project_dir, 'out')
        self.predict_model_dir = os.path.join(self.project_dir, 'out')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
