from src.base import Trainer
import torch.nn as nn
import transformers
from transformers import  DataCollatorWithPadding, get_scheduler
from datasets import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import evaluate
import pandas as pd
import json
import os
import logging


class CustomizableModel(nn.Module):
    def __init__(self, model_name = "xlm-roberta-base", num_labels=2):
        super(CustomizableModel, self).__init__()
        self.model = transformers.XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore # (loss), logits, (hidden_states), (attentions)
        return outputs
        

class ExtendedTrainer(Trainer):

    def __init__(self, dataset_path, sample_frac=1.0):
        super(ExtendedTrainer, self).__init__(dataset_path)
        self.model , self.tokenizer = self.load_model()
        self.train_data, self.eval_data = self.load_data(sample_frac=sample_frac)
        self.data_collator = self.get_data_collator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_data(self, sample_frac=1.0):
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
    
    def get_data_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def load_model(self):
        model = CustomizableModel()
        tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
        return model, tokenizer
    
    def train(
        self,
        output_dir, 
        learning_rate, 
        batch_size, 
        weight_decay, 
        epochs):
        logging.info(f"learning rate: {learning_rate}, batch size: {batch_size}, epochs: {epochs}")
        
        # dataloader
        train_dataloader = DataLoader(self.train_data, batch_size=batch_size, collate_fn=self.data_collator) # type: ignore
        test_dataloader = DataLoader(self.eval_data, batch_size=batch_size, collate_fn=self.data_collator) # type: ignore
        
        self.model = self.model.to(self.device)
        
        #define optimizer
        optimizer = transformers.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        #define step
        epochs = 3
        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        
        pbar = tqdm(range(num_training_steps))
        eval_pbar = tqdm(range(len(test_dataloader)))
        # train loop
        metric_output = None
        for epoch in range(epochs):
            metric = evaluate.load("accuracy")
            for batch in train_dataloader:
                self.model.train()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                self.model.zero_grad()
                pbar.update(1)
                if pbar.n % 100 == 0:
                    pbar.set_description(f"Epoch {epoch}: loss {loss.item():.3f}")
                
            metric = evaluate.load("accuracy")
            # evaluate
            self.model.eval()
            for batch in test_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
                eval_pbar.update(1)
            metric_output = metric.compute()
            # print results
            logging.info(f"Epoch {epoch}: {metric_output}")
            
            # save model
            # mkdir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(self.model.state_dict(), os.path.join(output_dir, f"model_{epoch}.pt"))
            with open(os.path.join(output_dir, f"model_{epoch}.json"), "w") as f:
                info_dict = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "accuracy": metric_output,
                    "loss": loss.item() # type: ignore
                }
                json.dump(info_dict, f, indent=4)
            self.last_checkpoint_path = os.path.join(output_dir, f"model_{epoch}.pt")

        return metric_output["accuracy"]