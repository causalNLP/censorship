import transformers
from transformers import DataCollatorWithPadding
import torch
import pandas as pd
from datasets import Dataset
import numpy as np
import evaluate
from shapley_scorer.train_hf import HF_Trainer


class CensorshipTrainer(HF_Trainer):
    def __init__(self, train_dataset_path, eval_dataset_path):
        super().__init__(train_dataset_path)
        self.eval_data_path = eval_dataset_path
    def load_data(self):
        train_data = pd.read_csv(self.dataset_path).dropna()
        eval_data = pd.read_csv(self.eval_data_path).dropna()
        
        self.train_data = Dataset.from_pandas(train_data)
        self.train_data = self.train_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        self.eval_data = Dataset.from_pandas(eval_data)
        self.eval_data = self.eval_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        
