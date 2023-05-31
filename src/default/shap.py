from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from src.base import ShapleyScorer
import transformers

import shap
from shap import maskers
import logging

class DefaultShapleyScorer(ShapleyScorer):
    def __init__(self, input_data_path:str,
                 output_data_path:str,
                 checkpoint_dir:str,
                 subset:float=0.001
        ):
        logging.warning("Warning: subset value is set to {}".format(subset))
        super(DefaultShapleyScorer, self).__init__(input_data_path, output_data_path, checkpoint_dir, subset)
        
    def load_model(self, checkpoint_dir:str):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        pipeline = transformers.pipeline('sentiment-analysis', model=model, tokenizer="xlm-roberta-base")
        return pipeline
    
    def get_explainer(self):
        masker = maskers.Text(mask_token="...") # mask token is used to mask words
        return shap.Explainer(self.model, masker)
    
