from src.default import DefaultShapleyScorer, DefaultTrainer
from src.extended import ExtendedShapleyScorer, ExtendedTrainer, CustomizableModel
from .base import Run, ShapleyScorer, Trainer

class ShapleyScorerFactory:
    @staticmethod
    def create(model_type:str, input_data_path:str, output_data_path:str, checkpoint:str,**args) -> ShapleyScorer:
        """_summary_
        Creates a shapley scorer object based on the model type.
        
        args:
        - model_type: str - either "default" or "extended"
        - input_data_path: str - path to the input data
        - output_data_path: str - path where the output data will be saved
        - checkpoint: str - for the default model, this is the folder containing the checkpoint or the name of the model on the hf hub,
                            for the extended model, this is the path to the checkpoint (.pt file)
        - (Optional) subset: float - fraction of the dataset to be used for scoring, default: 0.001
        """
        
        if model_type == "default":
            return DefaultShapleyScorer(
                input_data_path=input_data_path,
                output_data_path=output_data_path,
                checkpoint_dir=checkpoint,
                **args)
        elif model_type == "extended":
            return ExtendedShapleyScorer(
                input_data_path=input_data_path,
                output_data_path=output_data_path,
                checkpoint_path=checkpoint,
                **args)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))

class TrainerFactory:
    @staticmethod
    def create(model_type:str, dataset_path:str, sample_frac:float = 1.0, **args) -> Trainer:
        """
        Creates a trainer object based on the model type.
        
        args:
        - model_type: str - either "default" or "extended"
        - dataset_path: str - path to the dataset
        - (Optional) sample_frac: float - fraction of the dataset to be used for training 
        
        - For default trainer (optional):
            - model_name: str - name of the model to be used for sequence classification, default: "xlm-roberta-base"
        
        """
        if model_type == "default":
            return DefaultTrainer(dataset_path=dataset_path,
                                  sample_frac=sample_frac,
                                   **args)
        elif model_type == "extended":
            return ExtendedTrainer(dataset_path=dataset_path,
                                   sample_frac=sample_frac,
                **args)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))
    

