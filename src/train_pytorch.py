from src.model import Model
from src.dataloader import CensorDataset
from src.utils import count_parameters, Colors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, List
import logging
import json

from dataclasses import dataclass, field, asdict


@dataclass
class ConfigTrain:
    model_path: str
    num_of_gpus: int
    learning_rate: float
    num_of_epochs: int
    batch_size: int
    lang:str
    train_path: str
    eval_path: str


def load_dataset(config: ConfigTrain) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    # load the dataset
    train_dataset = CensorDataset(config.train_path, config.lang)
    eval_dataset = CensorDataset(config.eval_path, config.lang)

    # compute the weights for the loss function
    weights = train_dataset.get_weights()

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    logging.info("Data loaded")
    return train_loader, eval_loader, weights


def epoch_loop(dataloader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, epoch: int, weights: torch.Tensor) -> Tuple[float, float, float, float]:
    model.train()
    sum_loss = 0
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    for i, (tokens, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):

        tokens = tokens.to(0)
        labels = labels.to(0)
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
            logging.info(f"\n {Colors.INFO} epoch: {epoch}")
            logging.info(f"loss: {sum_loss / (i + 1)}")
            logging.info(f"accuracy: {sum_accuracy / (i + 1)}, {sum_precision/(i+1)}, {sum_recall/(i+1)} {Colors.END}")
            logging.info(f"confusion matrix: {confusion_matrix}")
            logging.info(f"{labels}, {pred}")

    return sum_loss / len(dataloader), sum_accuracy / len(dataloader), sum_precision / len(dataloader), sum_recall / len(dataloader)


def eval_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> Tuple[float, float, float, float]:
    model.eval()
    sum_loss = 0
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    with torch.no_grad():
        for i, (tokens, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):

            tokens = tokens.to(0)
            labels = labels.to(0)

            pred = model(tokens)

            loss = loss_fn(pred, labels)

            sum_loss += loss.item()
            accuracy, precision, recall, confusion_matrix = score(pred, labels)
            sum_accuracy += accuracy
            sum_precision += precision
            sum_recall += recall

    return sum_loss / len(dataloader), sum_accuracy / len(dataloader), sum_precision / len(dataloader), sum_recall / len(dataloader)


def score(pred: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float, List[List[int]]]:
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


def save_info_log(train_loss: float, train_accuracy: float, train_precision: float, train_recall: float, eval_loss: float, eval_accuracy: float, eval_precision: float, eval_recall: float, epoch: int, config: ConfigTrain) -> None:
    with open(config.model_path + "info.log", "a+") as f:
        f.write(f"------------------- Epoch {epoch} -------------------\n")
        f.write(f"Train loss: {train_loss}  Train accuracy: {train_accuracy}, Train precision: {train_precision}, Train recall: {train_recall}\n")
        f.write(f"Eval loss: {eval_loss}  Eval accuracy: {eval_accuracy}, Eval precision: {eval_precision}, Eval recall: {eval_recall}\n\n")


def train_pt(model_path: str,
             num_of_gpus: int,
             learning_rate: float,
             num_of_epochs: int,
             batch_size: int,
             lang:str,
             train_path: str,
             eval_path: str
    ) -> None:
    
    config = ConfigTrain(
        model_path=model_path,
        num_of_gpus=num_of_gpus,
        learning_rate=learning_rate,
        num_of_epochs=num_of_epochs,
        batch_size=batch_size,
        lang = lang,
        train_path=train_path,
        eval_path=eval_path
    )

    # save the config
    with open(config.model_path + ".json", "w") as f:
        config_dict = asdict(config)
        json.dump(config_dict, f, indent=4)

    logging.info(f"{Colors.INFO} start training")
    logging.info(f"Parameters: {config_dict}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logging.info(f"GPUS available: {torch.cuda.device_count()}")

    # load the dataset
    train_loader, eval_loader, weights = load_dataset(config)
    weights = weights.to(device)

    # create model
    model = Model()
    count_parameters(model)
    model = nn.DataParallel(model, device_ids=list(range(config.num_of_gpus)))
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # create loss function
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # train loop
    for epoch in range(config.num_of_epochs):

        epoch_loss, epoch_acc, epoch_precision, epoch_recall = epoch_loop(
            train_loader,
            model,
            optimizer,
            loss_fn,
            epoch,
            weights
        )

        logging.info(f"{Colors.INFO} Epoch {epoch+1} train loss: {epoch_loss} ")
        logging.info(f"Epoch {epoch+1} train acc: {epoch_acc}, train precision: {epoch_precision}, train recall: {epoch_recall} {Colors.ENDC}")

        eval_loss, eval_acc, eval_precision, eval_recall = eval_loop(
            eval_loader,
            model,
            loss_fn
        )

        logging.info(f"{Colors.INFO} Epoch {epoch+1} eval loss: {eval_loss}")
        logging.info(f"Epoch {epoch+1} eval acc: {eval_acc}, eval precision: {eval_precision},eval recall: {eval_recall} {Colors.ENDC}")

        save_info_log(
            epoch_loss,
            epoch_acc,
            epoch_precision,
            epoch_recall,
            eval_loss,
            eval_acc,
            eval_precision,
            eval_recall,
            epoch,
            config
        )
        torch.save(model, config.model_path + str(epoch) + ".pt")