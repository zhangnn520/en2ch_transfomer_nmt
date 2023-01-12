## A Transformer Framework Based Translation Task
### 一个基于Transformer网络结构的文本翻译模型

### 论文[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 基于PyTorch的实现

##### 声明：本工程源自https://github.com/moon-hotel/TransformerTranslation,本代码移植了NMT中一些代码，
#### 现在英译中bleu大于96%

## 1. 环境准备
```
matplotlib==3.5.3
numpy==1.20.0
packaging==21.3
pandas==1.3.5
sacrebleu==2.3.1
scipy==1.4.1
setuptools==65.5.0
torch==1.12.1+cu116
torchtext==0.6.0
tqdm==4.64.1
transformers==4.24.0
```

## 2. 使用方法
* STEP 1. 按照模型数据集格式进行改造数据集
* STEP 2. 可自定义修改配置文件`config.py`中的配置参数，也可以保持默认
### 2.1 训练
直接执行如下命令即可进行模型训练：
```
python train.py
```