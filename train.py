from tqdm import tqdm
from torch.optim import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from QA_Dataset import *

model_path = './model/model.pth'
TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 256   # 问题长度
T_LEN = 32    # 答案长度
BATCH_SIZE = 4
DEVICE = "cuda:0"

df = pd.read_csv("./data/SQuAD-v1.1.csv")

selected_columns = ["context", "question", "answer"]
df=df[:8000]

# Dataloader
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# 获取DataFrames的indices
train_indices = train_data.index.tolist()
val_indices = val_data.index.tolist()

# 使用indices随机取样
train_sampler = RandomSampler(train_indices)
val_sampler = RandomSampler(val_indices)

qa_dataset = QA_Dataset(TOKENIZER, df, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=20, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=20, sampler=val_sampler)

MODEL = MODEL.to(DEVICE)

train_loss = 0
val_loss = 0
train_batch_count = 0
val_batch_count = 0

for epoch in range(20):
    MODEL.train()
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        train_loss += outputs.loss.item()
        train_batch_count += 1

    # Evaluation
    MODEL.eval()
    for batch in tqdm(val_loader, desc="Validation batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        val_loss += outputs.loss.item()
        val_batch_count += 1

    print(
        f"{epoch + 1}/{20} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss / val_batch_count}")

torch.save(MODEL.state_dict(), model_path)