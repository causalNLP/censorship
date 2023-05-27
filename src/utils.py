import argparse
import matplotlib.pyplot as plt
import logging

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '-d', '--debug',
    help="Print lots of debugging statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose, print cuda device info and memory usage and training progress",
        action="store_const", dest="loglevel", const=logging.INFO,
    )

    parser.add_argument(
        '--batch-size', type=int,
        help="Set the batch size for each gpu",
        dest="batch_size",
        default=32,
    )
    parser.add_argument(
        '--epochs', type=int,
        help="Set the number of epochs",
        dest="epochs",
        default=10,
    )
    parser.add_argument(
        '--lr', type=float,
        help="Set the learning rate",
        dest="lr",
        default=1e-05,
    )
    parser.add_argument(
        '--lang', type=str,
        help="Set the language",
        dest="lang",
        default="tr",
    )

    parser.add_argument(
        '--max-len', type=int,
        help="Set the maximum length",
        dest="max_len",
        default=256,
    )
    parser.add_argument(
        '--GPUs', type=int,
        help="Set the number of GPUs",
        dest="GPUs",
        default=4,
    )
    parser.add_argument(
        '--model-path', type=str,
        help="Set the model path",
        dest="model_path"
    )


    args = parser.parse_args()
    return args

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