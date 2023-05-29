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
First, edit the `config.py` file to set the parameters of the model and the dataset.

Then, run the following command:
```bash
../shapley_scorer$ python run.py --config_path config.py --checkpoint_path /path/to/checkpoint <OPTION>
```
where:
- `--config_path` is the path to the config file
- `--checkpoint_path` is the path where save the checkpoint, or the path to the checkpoint to load for computing the Shapley values

The `<OPTION>` can be:
- `--finetune` to finetune the model on the dataset
- `--shap` to compute the Shapley values for each token of the dataset using the finetuned model
- `--hf_search` to search for the best hyperparameters. The hyperparameters find are saved in `<checkpoint_dir>/best_hyperparams.txt`
- `--all` to do all the above steps 


## Use a different dataset structure
The dataset is automatically splitted in train and test.
If you want to use a different dataset structure just override the `load_data()` method on the `HF_Trainer` class in `train_hf.py`. Note that the returned datasets must have this structure:

text | label |
-----|-------|
str  | int(0,1) |


