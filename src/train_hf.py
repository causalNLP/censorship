import transformers
from transformers import DataCollatorWithPadding
import torch
import pandas as pd
from datasets import Dataset
import numpy as np
import evaluate



def train_hf(
    output_dir:str,
    learning_rate:float,
    batch_size:int,
    epochs:int,
    train_dataset_path:str,
    eval_dataset_path:str,
):
    # import XLMR_BASE_ENCODER for sequence classification
    model = transformers.XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
    tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")

    data = pd.read_csv(train_dataset_path)
    eval_data = pd.read_csv(eval_dataset_path)
    eval_data = eval_data[["text_proc", "censored"]].dropna()
    eval_data.columns = ["text", "labels"]
    data = data[["text_proc", "censored"]].dropna()
    data.columns = ["text", "labels"]

    train = Dataset.from_pandas(data)
    train = train.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
    eval = Dataset.from_pandas(eval_data)
    eval = eval.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

    data_collar = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = transformers.TrainingArguments(
        output_dir= output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = transformers.Trainer(
        model = model,
        args = training_args,
        train_dataset = train,
        eval_dataset = eval,
        tokenizer=tokenizer,
        data_collator=data_collar,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()