from src.model import  Model, Config
from src.dataloader import CensorDataset
from src.utils import get_args, count_parameters

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Dict
import logging
import json

args = get_args()
logging.basicConfig(level=args.loglevel)

@dataclass
class ConfigTrain:
    number_of_gpus:int = args.GPUs
    batch_size:int = args.batch_size * number_of_gpus
    epochs:int = args.epochs
    lr:float = args.lr
    lang:str = args.lang
    max_len:int = args.max_len
    model_path:str = args.model_path
    train_path:str = "data/dataset_train.csv"
    eval_path:str = "data/dataset_val.csv"
    device:str = "cuda:0"
    use_hf_transformers:bool = args.hf_model
    model_config: Dict = field(default_factory=lambda: Config().asdict())


    def asdict(self):
        dicts = {
            "number_of_gpus": self.number_of_gpus,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "lang": self.lang,
            "max_len": self.max_len,
            "model_path": self.model_path,
            "train_path": self.train_path,
            "eval_path": self.eval_path,
            "device": self.device,
            "model_config": self.model_config
        }
        return dicts
    def __repr__(self) -> str:
        return f"""
        batch_size: {self.batch_size}
        epochs: {self.epochs}
        lr: {self.lr}
        lang: {self.lang}
        train_path: {self.train_path}
        eval_path: {self.eval_path}
        """




def load_dataset():
    # load the dataset
    train_dataset = CensorDataset(ConfigTrain.train_path)
    eval_dataset = CensorDataset(ConfigTrain.eval_path)

    # compute the weights for the loss function
    weights = train_dataset.get_weights()
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=ConfigTrain.batch_size,  shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=ConfigTrain.batch_size, shuffle=True, drop_last=True)
    logging.info("Data loaded")
    return train_loader, eval_loader, weights


def epoch_loop(dataloader, model, optimizer, loss_fn, epoch, weights):
    model.train() 
    sum_loss = 0
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    for i, (tokens, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):

        tokens = tokens.to(ConfigTrain.device)
        labels = labels.to(ConfigTrain.device)
        optimizer.zero_grad()

        pred = model(tokens)
        loss = loss_fn(pred, labels)
           
        loss.backward(retain_graph=True)
        
        optimizer.step()

        sum_loss += loss.item()
        accuracy, precision, recall, confusion_matrix = score(pred, labels)
        sum_accuracy += accuracy
        sum_precision += precision
        sum_recall += recall
        
        if i % 100 == 0:
            # print loss
            logging.info(f"\nepoch: {epoch}")
            logging.info(f"loss: {sum_loss / (i + 1)}")
            logging.info(f"accuracy: {sum_accuracy / (i + 1)}, {sum_precision/(i+1)}, {sum_recall/(i+1)}")
            logging.info(f"confusion matrix: {confusion_matrix}")
            logging.info(f"{labels}, {pred}")

    return sum_loss / len(dataloader), sum_accuracy / len(dataloader), sum_precision / len(dataloader), sum_recall / len(dataloader)

def eval_loop(dataloader, model, loss_fn):
    model.eval()
    sum_loss = 0
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    for i, (tokens, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):

        tokens = tokens.to(ConfigTrain.device)
        labels = labels.to(ConfigTrain.device)

        pred = model(tokens)

        loss = loss_fn(pred, labels)
        
        sum_loss += loss.item()
        accuracy, precision, recall, confusion_matrix = score(pred, labels)
        sum_accuracy += accuracy
        sum_precision += precision
        sum_recall += recall

    return sum_loss / len(dataloader), sum_accuracy / len(dataloader), sum_precision / len(dataloader), sum_recall / len(dataloader)

def score(pred, labels):
    # positive outcome is the 0 class
    tp = ((pred.argmax(1) == labels) & (labels == 0)).sum().item()
    tn = ((pred.argmax(1) == labels) & (labels == 1)).sum().item()
    fp = ((pred.argmax(1) != labels) & (labels == 0)).sum().item()
    fn = ((pred.argmax(1) != labels) & (labels == 1)).sum().item()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = 0 if tp == 0 else tp / (tp + fp)
    recall = 0 if tp == 0 else tp / (tp + fn)
    confusion_matrix = [[tp, fp], [fn, tn]]
    return accuracy, precision, recall, confusion_matrix
    

    
def save_info_log(
            train_loss, 
            train_accuracy,
            train_precision,
            train_recall, 
            eval_loss, 
            eval_accuracy,
            eval_precision,
            eval_recall, 
            epoch):
    with open(ConfigTrain.model_path + "info.log", "a+") as f:
        f.write(f"------------------- Epoch {epoch} -------------------\n")
        f.write(f"Train loss: {train_loss}  Train accuracy: {train_accuracy}, Train precision: {train_precision}, Train recall: {train_recall}\n")
        f.write(f"Eval loss: {eval_loss}  Eval accuracy: {eval_accuracy}, Eval precision: {eval_precision}, Eval recall: {eval_recall}\n\n")


def train():
    # save the config
    with open(ConfigTrain.model_path + str(0) + ".json", "w") as f:
            json.dump(ConfigTrain().asdict(), f, indent=4)
            
    logging.info("start training")
    
    if torch.cuda.is_available():
        logging.info(f"GPUS available: {ConfigTrain.number_of_gpus}")
        
    
    # load the dataset
    train_loader, eval_loader, weights = load_dataset()
    weights = weights.to(ConfigTrain.device) 

    # create model
    model = Model()
    count_parameters(model)
    # model = torch.compile(model)
    model = nn.DataParallel(model, device_ids=list(range(ConfigTrain.number_of_gpus)))
    model.to(ConfigTrain.device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ConfigTrain.lr)
    

    print(weights)
    # create loss function
    loss_fn = nn.CrossEntropyLoss(weight = weights)

    # train loop
    for epoch in range(ConfigTrain.epochs):

        epoch_loss, epoch_acc, epoch_precision, epoch_recall = epoch_loop(
                                                                    train_loader,
                                                                    model, 
                                                                    optimizer, 
                                                                    loss_fn, 
                                                                    epoch, 
                                                                    weights
        )

        logging.info(f"Epoch {epoch+1} train loss: {epoch_loss}")
        logging.info(f"Epoch {epoch+1} train acc: {epoch_acc}, train precision: {epoch_precision}, train recall: {epoch_recall}")
        

        
        eval_loss, eval_acc, eval_precision, eval_recall = eval_loop(
                                eval_loader,
                                model, 
                                loss_fn
        )

        logging.info(f"Epoch {epoch+1} eval loss: {eval_loss}")
        logging.info(f"Epoch {epoch+1} eval acc: {eval_acc}, eval precision: {eval_precision},eval recall: {eval_recall}")        
        
        save_info_log(
            epoch_loss,
            epoch_acc,
            epoch_precision,
            epoch_recall,
            eval_loss,
            eval_acc,
            eval_precision,
            eval_recall,
            epoch
        )
        torch.save(model.module.state_dict(), ConfigTrain.model_path + str(epoch) + ".pt")


import transformers
import torch
import pandas as pd

def train_hf_transformers():
    # import XLMR_BASE_ENCODER for sequence classification
    model = transformers.XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
    tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")

    data = pd.read_csv("data/dataset_train.csv")
    eval_data = pd.read_csv("data/dataset_val.csv")
    eval_data = eval_data[["text_proc", "censored"]].dropna()
    eval_data.columns = ["text", "labels"]
    data = data[["text_proc", "censored"]].dropna()
    data.columns = ["text", "labels"]

    from datasets import Dataset
    train = Dataset.from_pandas(data)
    train = train.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
    eval = Dataset.from_pandas(eval_data)
    eval = eval.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)

    from transformers import DataCollatorWithPadding
    data_collar = DataCollatorWithPadding(tokenizer=tokenizer)

    import evaluate
    accuracy = evaluate.load("accuracy")

    import numpy as np
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = transformers.TrainingArguments(
        output_dir= "HF_" + ConfigTrain.model_path,
        learning_rate=ConfigTrain.lr,
        per_device_train_batch_size=ConfigTrain.batch_size,
        per_device_eval_batch_size=ConfigTrain.batch_size,
        num_train_epochs=ConfigTrain.epochs,
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
    



if __name__ == "__main__":
    if ConfigTrain.use_hf_transformers:
        logging.info("Training with HF transformers")
        train_hf_transformers()
    else:
        logging.info("Training with custom model")
        train()