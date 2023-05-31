from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from src.base import ShapleyScorer
from src.extended.trainer import CustomizableModel
import transformers
import numpy as np
import scipy as sp
import torch
import shap
from shap import maskers
import re
import logging

class ExtendedShapleyScorer(ShapleyScorer):
    def __init__(self, input_data_path:str,
                 output_data_path:str,
                 checkpoint_path:str,
                 subset_frac:float=0.0001
        ):
        logging.warning(f"WARNING: subset frac is set to {subset_frac}")
        super(ExtendedShapleyScorer, self).__init__(input_data_path, output_data_path, checkpoint_path, subset_frac)
        
    def load_model(self, checkpoint_path:str):
        tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = CustomizableModel()
        model.load_state_dict(torch.load(checkpoint_path))
        model = model.to(self.device)
        device = self.device
        
        def f(x):
            tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).to(device)
            attention_mask = (tv!=0).type(torch.int64).to(device)
            outputs = model(tv,attention_mask=attention_mask)[0].detach().cpu().numpy()
            scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
            val = sp.special.logit(scores)
            return val
            
        return f
        
    def get_explainer(self):
        def custom_tokenizer(s, return_offsets_mapping=True):
            """ Custom tokenizers conform to a subset of the transformers API.
            """
            pos = 0
            offset_ranges = []
            input_ids = []
            for m in re.finditer(r"\W", s):
                start, end = m.span(0)
                offset_ranges.append((pos, start))
                input_ids.append(s[pos:start])
                pos = end
            if pos != len(s):
                offset_ranges.append((pos, len(s)))
                input_ids.append(s[pos:])
            out = {}
            out["input_ids"] = input_ids
            if return_offsets_mapping:
                out["offset_mapping"] = offset_ranges
            return out

        return shap.Explainer(self.model, masker=maskers.Text(custom_tokenizer))