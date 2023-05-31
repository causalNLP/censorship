import transformers
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.models.roberta.modeling_roberta import SequenceClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset
import numpy as np
import evaluate
from shapley_scorer.train_hf import HF_Trainer

# Set colors for terminal output
STEP = "\033[92m"
INFO = "\033[93m"
ENDC = "\033[0m"

class CensorshipTrainer(HF_Trainer):
    def __init__(self, train_dataset_path, eval_dataset_path):
        super().__init__(train_dataset_path)
        self.model = CustomModel(num_labels=2)
        self.eval_data_path = eval_dataset_path
    def load_data(self):
        train_data = pd.read_csv(self.dataset_path)
        train_data = train_data[["text_proc", "censored"]].dropna()
        train_data.columns = ["text", "label"]
        eval_data = pd.read_csv(self.eval_data_path)
        eval_data = eval_data[["text_proc", "censored"]].dropna()
        eval_data.columns = ["text", "label"]
        print("dataset length: ", len(train_data))
        self.train_data = Dataset.from_pandas(train_data)
        self.train_data = self.train_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        self.eval_data = Dataset.from_pandas(eval_data)
        self.eval_data = self.eval_data.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
    
    def train(self,
              output_dir,
              batch_size,
              weight_decay,
              learning_rate,
              epochs):
        raise NotImplementedError("Custom training loop not implemented yet")
        
        

class CustomModel(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.model = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-base", num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
        