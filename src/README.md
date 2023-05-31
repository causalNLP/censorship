# Finetune a model and compute shapely scores

This folder contains the code to finetune a model and compute shapely scores. 
## Dataset
The input dataset is a csv file with the following columns:
- `text`: the text to classify
- `label`: the label of the text (int)

## High level usage
The file `run.py` contains a high level usage of the code. Modify the `config` class to change the parameters of the finetuning and the shapley computation.
To run the code:
```bash
$ python run.py [--finetune or --shap] --last-checkpoint <path_to_checkpoint> 
```

## Default models from huggingface hub
The folder `default` contain the code to finetune a model from the huggingface hub using the HF trainer API.

The class `DefaultTrainer` is initialized with the following arguments:
- `dataset_path`: path to the dataset (csv file)
- `model_name`: name of the model to finetune or path to the model (huggingface compatible)

The class `DefaultShapleyScorer` is initialized with the following arguments:
- `input_data_path`: path to the dataset (csv file)
- `output_data_path`: path to the output file (csv file)
- `checkpoint_path`: path to the checkpoint of the model to use or name of the pretrained model (huggingface compatible)

## Extend the models
The folder `extend` contain the code to finetune a model from the huggingface hub using the HF trainer API and extend it with a new head.
The class `CustomizableModel` is the base class for all the models supported by this trainer.
The `ExtendedTrainer` class implement a custom training loop to finetune the model and the new head, it is initialized with the following arguments:
- `dataset_path`: path to the dataset (csv file)

The class `ExtendedShapleyScorer` is initialized with the following arguments:
- `input_data_path`: path to the dataset (csv file)
- `output_data_path`: path to the output file (csv file)
- `checkpoint_path`: path to the checkpoint (.pt) file of the model to use or name of the pretrained model (of class `CustomizableModel`)
The class implement a custom model for the shapley computation and a custom tokenizer to mask the token during the computation.

## Usage
To finetune a model initialize the trainer class and call the `train` method,
To compute the shapley scores initialize the scorer class and call the `run_shap` method, then call the `shap_values_for_words` to compute the average of the shap score for each word and `save_shap_values` to save the results in a csv file.

## To do
- [ ] Add method to search for the best hyperparameters.