from transformers import EarlyStoppingCallback
import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset
from thesis.connect_sql import connect_sql
from sklearn.metrics import accuracy_score, f1_score


def main():
    # 固定随机种子
    set_seed(42)

    # 连接数据库
    conn = connect_sql.Connect()
    sql = "SELECT id, text, label FROM Training_Set"
    df = pd.read_sql(sql, conn)
    conn.close()

    # label 转整数
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    if df["label"].dtype == object:
        df["label"] = df["label"].map(label_map)

    # 转 Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # 加载 tokenizer & model
    model_name = "uer/roberta-base-finetuned-dianping-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # Tokenize + 并行预处理
    def preprocess(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    dataset = dataset.map(
        preprocess,
        batched=True,
        num_proc=10,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )

    # 划分训练/验证集
    dataset = dataset.train_test_split(test_size=0.1)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 训练参数优化
    training_args = TrainingArguments(
        output_dir="./tourism_sentiment_model",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none"
    )

    # 计算指标
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1": f1}

    # Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 开始训练
    torch.cuda.empty_cache()
    trainer.train()

    # 保存模型
    trainer.save_model("./tourism_sentiment_model")
    tokenizer.save_pretrained("./tourism_sentiment_model")

    # 推理示例
    texts = ["景色非常漂亮，服务态度很好", "景区人太多了，体验不好"]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).tolist()
    print("预测结果：", preds)


if __name__ == "__main__":
    main()
