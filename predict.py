import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config
import warnings

warnings.filterwarnings("ignore")

DEVICE = "cpu"
model_path = './model/model.pth'

# 加载分词器和模型配置
TOKENIZER = T5TokenizerFast.from_pretrained("./t5-base")
config = T5Config.from_pretrained("./t5-base")

# 初始化模型
MODEL = T5ForConditionalGeneration(config)

# 加载 .pth 文件中的模型参数
MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
MODEL = MODEL.to(DEVICE)

Q_LEN = 256   # 问题长度
T_LEN = 32    # 答案长度

def predict_answer(context, question):
    inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)
    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)

    return predicted_answer

if __name__ == '__main__':
    context = "My name is Sarah and I live in London"
    question = "Where do I live?"
    print(predict_answer(context, question))
