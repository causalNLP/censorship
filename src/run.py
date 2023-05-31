from src import ShapleyScorerFactory, TrainerFactory
from dataclasses import dataclass
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO)
args = argparse.ArgumentParser()

args.add_argument("--finetune", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Fine tuning the model", dest="finetuning")
args.add_argument("--shap" , type=bool, action=argparse.BooleanOptionalAction, default=False, help="Shapley values", dest="shap")
args.add_argument("--last-checkpoint", type=str,  help="Last checkpoint", dest="last_checkpoint")

args = args.parse_args()
@dataclass
class FinetuningConfig():
    model_type:str = "default"
    dataset_path:str = "data/dataset.csv"
    checkpoint:str = "checkpoints"
    batch_size:int = 2
    learning_rate:float = 1e-5
    weight_decay:float = 0.01
    epochs:int = 1
    
@dataclass
class ShapConfig():
    model_type:str = "default"
    dataset_path:str = "data/dataset.csv"
    checkpoint:str = "checkpoints"
    output_shaply_path:str = "data/shapley_values.csv"
    subset_shapeley:float = 0.0001
    subset_dataset:float = 0.0001
    
@dataclass
class HyperparameterConfig():
    model_type:str = "default"
    dataset_path:str = "data/dataset.csv"
    checkpoint_folder:str = "checkpoints"
    epochs:int = 3
    subset_dataset:float = 0.5


def finetuning():
    config = FinetuningConfig()
    trainer = TrainerFactory.create(config.model_type,
                                    dataset_path = config.dataset_path,
                                    sample_frac=config.subset_dataset
                                    )
    trainer.train(output_dir=config.checkpoint,
                  learning_rate=config.learning_rate,
                  batch_size=config.batch_size,
                  weight_decay=config.weight_decay,
                  epochs=config.epochs
    )


def shapeley(last_checkpoint_path:str=None):
    

    config = ShapConfig()
    scorer = ShapleyScorerFactory.create(config.model_type,
                                         input_data_path = config.dataset_path,
                                         output_data_path=config.output_shaply_path,
                                         checkpoint=last_checkpoint_path,
                                         subset=config.subset_shapeley)
    label = "LABEL_1" if config.model_type == "default" else  1
    scorer.run_shap(
        label_value=label,
    )
    scorer.shap_values_for_words()
    scorer.save_shap_values()

def hyperparameter_search():
    config = HyperparameterConfig()
    trainer = TrainerFactory.create(config.model_type,
                                    dataset_path = config.dataset_path,
                                    sample_frac=config.subset_dataset)
    best_param, best_accuracy = trainer.hyperparameter_search(
        output_dir=config.checkpoint,
        epochs = config.epochs
    )
    logging.info(f"Best accuracy: {best_accuracy}")
    with open(os.path.join(config.checkpoint_folder, "best_param.txt"), "w") as f:
        f.write(str(best_param))

if __name__ == "__main__":

    if args.finetuning:
        finetuning()
    elif args.shap:
        print(args.last_checkpoint)
        shapeley(args.last_checkpoint)

    
    
    
    