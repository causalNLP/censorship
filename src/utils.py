import matplotlib.pyplot as plt
import logging

class Colors:
    INFO = "\033[94m"
    DEBUG = "\033[92m"
    SMALL_INFO = "\033[96m"
    END = "\033[0m"

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    logging.info(table)
    logging.info(f"Total Trainable Params: {total_params}")
    return total_params

def plot_loss_across_epochs(total_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(total_loss, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(ConfigTrain.model_path + "loss.png")
    
    
