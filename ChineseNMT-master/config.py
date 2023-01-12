import torch

best_bleu_score = 96.80
d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 5000
tgt_vocab_size = 5000
batch_size = 256
epoch_num = 30
early_stop = 1000
lr = 2e-5

# greed decode的最大句子长度
max_len = 60
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True
model_init = False  # 是否需要从头进行训练模型，反之加载训练好的模型
pretrain_model_path = './experiment/model1dev_loss_0.027935555204749107_bleu_96.83944922382534.pth'  # 加载训练好的模型
test_model_path = './experiment/model1dev_loss_0.027935555204749107_bleu_96.83944922382534.pth'  # 加载预测模型
data_dir = './data'
train_data_path = 'data/enoch_json/train.json'
dev_data_path = 'data/enoch_json/dev.json'
test_data_path = 'data/enoch_json/test.json'
model_path = './experiment/model'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'
predict_path = './experiment/predict.json'
# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
