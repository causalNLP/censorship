# Finetune a RoBERTa model and compute Shapley values for each token

## Installation

Requirements:
- Python 3.6+
- shap
- transformers
- datasets
- accelerate
- torch

## Usage
First, edit the `config.json` file to set the parameters of the model and the dataset.

Then, run the following command:
```bash
$ python run.py --config_path config.py --checkpoint_path /path/to/checkpoint <OPTION>
```
where:
- `--config_path` is the path to the config file
- `--checkpoint_path` is the path to the checkpoint to load for computing the Shapley values

The `<OPTION>` can be:
- `--shap` to compute the Shapley values for each token of the dataset using the finetuned model
- `--finetune` to finetune the model on the dataset using RoBerta base model
- `--hf_search` to search for the best hyperparameters. The hyperparameters find are saved in `<checkpoint_dir>/best_hyperparams.txt`
- `--all` to do all the above steps, hf_search, finetune, shap values (in this case the `--checkpoint_path` is not needed)

## Structure of config.json
```json
{
    "global": {
        "dataset_path": "<path to dataset>",
        "checkpoint_dir": "<path to checkpoint dir>", #used only if --checkpoint_path is not specified for computing shapley values
    },

    "hyperparameter_search": {
        "num_trials": <int: number of different random trials>,
        "output_dir": "<folder where save the results of the search>",
        "batch_size": <int: batch size for the search>,
        "search_space" : {
            "learning_rate": [1e-6, 1e-5, 1e-4, 1e-3],
            "num_epochs": [1],
            "weight_decay": [0.0, 0.1, 0.2, 0.3]
        }
    },

    "finetuning": {
        "batch_size": 32,
        "weight_decay": 0.01,
        "epochs": 10,
        "learning_rate": 0.0001,
        "output_dir": "<folder where save the finetuned model>" 
    },

    "shapley": {
        "output_data_path": "<path where save the csv with the shapley values>",
        "data_subset": 0.0001 #percentage of the dataset to use for computing the shapley values
    }
}
```

## Use a different dataset structure
The dataset is automatically splitted in train and test.
If you want to use a different dataset structure just override the `load_data()` method on the `HF_Trainer` class in `train_hf.py`. Note that the returned datasets must have this structure:

text | label |
-----|-------|
str  | int(0,1) |


