import logging
import argparse
from src.utils import count_parameters, Colors
from src.train_pytorch import train_pt
from src.train_hf import CensorshipTrainer


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
        default=None,
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
    parser.add_argument(
        "--hf-model", default=False,
        action = "store_true",
        help="Set the huggingface model",
        dest="hf_model",
    )
    parser.add_argument(
        "--train-dataset-path", type=str,
        default="data/dataset_train.csv",
        help="Set the train dataset path",
        dest="train_dataset_path",
    )
    parser.add_argument(
        "--eval-dataset-path", type=str,
        default="data/dataset_val.csv",
        help="Set the eval dataset path",
        dest = "eval_dataset_path",
    )

    args = parser.parse_args()
    return args

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    args = get_args()
    if args.hf_model:
        logging.info(f"{Colors.INFO} Using HuggingFace model {Colors.END}")
        trainer = CensorshipTrainer( args.train_dataset_path, args.eval_dataset_path)
        trainer.load_data()
        trainer.train(
            output_dir=args.model_path,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            weight_decay=0.01,
        )
        
    else:
        logging.info(f"{Colors.INFO} Using PyTorch model {Colors.END}")
        train_pt(
            model_path=args.model_path,
            num_of_gpus=args.GPUs,
            learning_rate=args.lr,
            num_of_epochs=args.epochs,
            batch_size=args.batch_size,
            lang=args.lang,
            train_path=args.train_dataset_path,
            eval_path=args.eval_dataset_path
        )
    
if __name__ == "__main__":
    main()