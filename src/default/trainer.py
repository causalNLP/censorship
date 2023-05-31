from src.base import Trainer
import transformers
import evaluate
import numpy as np
import os
import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import logging


class DefaultTrainer(Trainer):
    def __init__(self, dataset_path, sample_frac:float = 1.0, model_name="xlm-roberta-base"):
        super(DefaultTrainer, self).__init__(dataset_path)
        self.model_name = model_name
        self.model , self.tokenizer = self.load_model()
        self.train_data, self.eval_data = self.load_data(sample_frac= sample_frac)
        self.data_collator = self.get_data_collator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_data(self, sample_frac=1.0) -> Tuple[Dataset, Dataset]:
        """
        Load data from dataset_path and split into train and eval
        """

        data = pd.read_csv(self.dataset_path)
        data = data.sample(frac=sample_frac, random_state=0)
        data = data[["text", "label"]].dropna()
        # split data into train and eval
        train = data.sample(frac=0.8, random_state=0)
        eval = data.drop(train.index)

        train_data = Dataset.from_pandas(train)
        train_data = train_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        eval_data = Dataset.from_pandas(eval)
        eval_data = eval_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        
        train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        eval_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        
        return train_data, eval_data
    
    def load_model(self):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
    
    def get_data_collator(self):
        return transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def train(self,
              output_dir, 
              learning_rate, 
              batch_size, 
              weight_decay, 
              epochs):
        """
        Train model with given hyperparameters and save to output_dir
        """

        
        logging.info(f"Learning rate: {learning_rate}, batch size: {batch_size}, epochs: {epochs}")
        

        training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            optim="adamw_torch"
        )


        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        logging.info("Start training")
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data, # type: ignore
            eval_dataset=self.eval_data, # type: ignore
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics # type: ignore
        )

        
        trainer.train()
        eval_metrics = trainer.evaluate()
        
        last_checkpoint_path = None
        # find folder of last checkpoint
        last_checkpoint_folder = None
        for folder in os.listdir(output_dir):
            if folder.startswith("checkpoint-"):
                last_checkpoint_folder = folder
                
        if last_checkpoint_folder is not None:
            last_checkpoint_path = os.path.join(output_dir, last_checkpoint_folder)
            logging.info(f"Model saved in: {last_checkpoint_path}")
        self.last_checkpoint_path = last_checkpoint_path
        return eval_metrics["eval_accuracy"]

