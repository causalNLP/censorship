import argparse
import json
import os

from shapley_score import ShapleyScorer
from train_hf import HF_Trainer


# Set colors for terminal output
STEP = "\033[94m"
INFO = "\033[92m"
ENDC = "\033[0m"
WARNING = "\033[93m"


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, default="config.json")
    argparser.add_argument("--hp_search", action="store_true")
    argparser.add_argument("--finetune", action="store_true")
    argparser.add_argument("--shap", action="store_true")
    argparser.add_argument("--all", action="store_true")
    argparser.add_argument("--checkpoint_dir", type=str, default=None)
    return argparser.parse_args()


def parse_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def run_hyperparameter_search(config):
    print(f"{STEP} Running hyperparameter search {ENDC}")
    trainer = HF_Trainer(config["global"]["dataset_path"])
    trainer.load_data()
    best_hyperparameters = trainer.hyperparameter_search(
        num_trials=config["hyperparameter_search"]["num_trials"],
        output_dir=config["hyperparameter_search"]["output_dir"],
        batch_size=config["hyperparameter_search"]["batch_size"],
        search_space=config["hyperparameter_search"]["search_space"],
    )
    return best_hyperparameters


def run_finetuning(config):
    print(f"{STEP} Running finetuning {ENDC}")
    trainer = HF_Trainer(config["global"]["dataset_path"])
    trainer.load_data()
    trainer.train(
        output_dir=config["finetuning"]["output_dir"],
        learning_rate=config["finetuning"]["learning_rate"],
        batch_size=config["finetuning"]["batch_size"],
        weight_decay=config["finetuning"]["weight_decay"],
        epochs=config["finetuning"]["epochs"],
    )
    last_checkpoint_folder = trainer.last_checkpoint_path
    return last_checkpoint_folder


def run_shap(config):
    print(f"{STEP} Running shapley scorer {ENDC}")
    scorer = ShapleyScorer(
        input_data_path=config["global"]["dataset_path"],
        output_data_path=config["shapley"]["output_data_path"]
    )
    scorer.pipeline(
        model_checkpoint=config["global"]["checkpoint_dir"],
        data_subset=config["shapley"]["data_subset"],
    )


def main():
    args = parse_args()
    config = parse_config(args.config_path)
    if args.checkpoint_dir is not None:
        config["global"]["checkpoint_dir"] = args.checkpoint_dir

    if args.hp_search:
        run_hyperparameter_search(config)
    elif args.finetune:
        run_finetuning(config)
    elif args.shap:
        run_shap(config)
    elif args.all:
        best_hyperparameters = run_hyperparameter_search(config)
        config["finetuning"]["learning_rate"] = best_hyperparameters["learning_rate"]
        config["finetuning"]["epochs"] = best_hyperparameters["num_epochs"]
        config["finetuning"]["weight_decay"] = best_hyperparameters["weight_decay"]
        last_checkpoint_folder = run_finetuning(config)
        config["global"]["checkpoint_dir"] = last_checkpoint_folder
        run_shap(config)
    else:
        print(f"{WARNING} No arguments provided, exiting. Please provide at least one of the following arguments: --hp_search, --finetune, --shap, --all{ENDC}")
        exit(0)


if __name__ == "__main__":
    main()