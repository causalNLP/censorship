from shapley_score import ShapleyScorer
from train_hf import train_hf
import argparse
from train_hf import HF_Trainer
argparser = argparse.ArgumentParser()
argparser.add_argument("--config_path", type=str, default="config.json", dest="config_path")
argparser.add_argument("--run_hyperparameter_search", default=False, action="store_true", dest="run_hyperparameter_search")
argparser.add_argument("--run_finetuning", default=False, action="store_true", dest="run_finetuning")
argparser.add_argument("--run_shap", default=False, action="store_true", dest="run_shap")
argparser.add_argument("--run_all", default=False, action="store_true", dest="run_all")
argparser.add_argument("--checkpoint_path", type=str, default="output", dest="checkpoint_path")
args = argparser.parse_args()


def parse_config(config_path:str = "config.json"):
    import json
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def run_hyperparameter_search(config):
    trainer = HF_Trainer(config["global"]["dataset_path"])
    trainer.load_data()
    best_hyperparameters = trainer.hyperparameter_search(
        num_trials=config["hyperparameter_search"]["num_trials"],
        output_dir=config["hyperparameter_search"]["output_dir"],
        batch_size=config["hyperparameter_search"]["batch_size"],
    )
    return best_hyperparameters
    
def run_finetuning(config):
    trainer = HF_Trainer(config["global"]["dataset_path"])
    trainer.load_data()
    trainer.train(
        output_dir=config["global"]["checkpoint_path"],
        learning_rate=config["finetuning"]["learning_rate"],
        batch_size=config["finetuning"]["batch_size"],
        weight_decay=config["finetuning"]["weight_decay"],
        epochs=config["finetuning"]["epochs"],
    )
    
def run_shap(config):
    scorer = ShapleyScorer(
        input_data_path=config["global"]["dataset_path"],
        output_data_path=config["shapley"]["output_data_path"]
    )
    scorer.pipeline(
        model_type = "xlm-roberta-base",
        model_checkpoint=config["global"]["checkpoint_path"]
    )
    

        

if __name__ == "__main__":
    config = parse_config(config_path = args.config_path)
    config["global"]["checkpoint_path"] = args.checkpoint_path
    
    if args.run_hyperparameter_search:
        run_hyperparameter_search(config)
    if args.run_finetuning:
        run_finetuning(config)
    if args.run_shap:
        run_shap(config)
    if args.run_all:
        best_hyperparameters = run_hyperparameter_search(config)
        config["finetuning"]["learning_rate"] = best_hyperparameters["learning_rate"]
        config["finetuning"]["batch_size"] = best_hyperparameters["batch_size"]
        config["finetuning"]["weight_decay"] = best_hyperparameters["weight_decay"]
        run_finetuning(config)
        run_shap(config)
    else:
        print("No arguments provided, exiting. Please provide at least one of the following arguments: --run_hyperparameter_search, --run_finetuning, --run_shap, --run_all")
        exit(0)