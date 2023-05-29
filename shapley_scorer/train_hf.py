import transformers
from transformers import DataCollatorWithPadding
import torch
import pandas as pd
from datasets import Dataset
import numpy as np
import evaluate
import random


class HF_Trainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = transformers.XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.accuracy = evaluate.load("accuracy")

    def load_data(self):
        data = pd.read_csv(self.dataset_path)
        data = data[["text", "label"]].dropna()
        # split data into train and eval
        train = data.sample(frac=0.8, random_state=0)
        eval = data.drop(train.index)

        self.train_data = Dataset.from_pandas(train)
        self.train_data = self.train_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        self.eval_data = Dataset.from_pandas(eval)
        self.eval_data = self.eval_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

    def train(self,
              output_dir, 
              learning_rate, 
              batch_size, 
              weight_decay, 
              epochs):
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return self.accuracy.compute(predictions=predictions, references=labels)

        training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )

        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.evaluate()

    def hyperparameter_search(self, 
                              num_trials, 
                              output_dir, 
                              batch_size):
        search_space = {
            "learning_rate": [1e-6, 1e-5, 1e-4, 1e-3],
            "num_epochs": [1, 2, 3, 4, 5],
            "weight_decay": [0.0, 0.1, 0.2, 0.3]
        }
        self.load_data()
        best_hyperparameters = {}
        best_metric = 0
        for i in range(num_trials):
            hyperparameters = {
                "learning_rate": random.choice(search_space["learning_rate"]),
                "num_epochs": random.choice(search_space["num_epochs"]),
                "weight_decay": random.choice(search_space["weight_decay"])
            }
            trial_output_dir = f"{output_dir}/trial_{i}"
            self.train(
                output_dir=trial_output_dir,
                learning_rate=hyperparameters["learning_rate"],
                batch_size=batch_size,
                weight_decay=hyperparameters["weight_decay"],
                epochs=hyperparameters["num_epochs"]
            )
            # evaluate the model
            eval_results = transformers.EvaluationResults.from_files(trial_output_dir)
            metric = eval_results.metrics["eval_accuracy"]

            if metric > best_metric:
                best_metric = metric
                best_hyperparameters = hyperparameters

        print(f"Best hyperparameters: {best_hyperparameters}")
        print(f"Best metric: {best_metric}")
        return best_hyperparameters