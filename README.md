# 文件说明
- data: 数据集
- model: 存放训练好的.pth文件
- t5-base: 存放t5预训练模型
  - config.json 定义模型的结构和超参数。
  - model.safetensors 预训练模型权重参数
  - spiece.model 用于分词的SentencePiece模型文件
  - tokenizer.json 词汇表

> t5-base 文件是由 https://hf-mirror.com/google-t5/t5-base/tree/main 下载到本地的

# 训练
https://www.kaggle.com/code/discodzy/t5-squad/script
在kaggle上训练，得到.pth文件
# 预测
将训练得到的.pth文件放入model文件夹下
`python predict.py`