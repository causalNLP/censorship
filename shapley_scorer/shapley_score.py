import pandas as pd
import transformers
import torch
import re
import shap
from tqdm import tqdm
import concurrent.futures
import numpy as np

    
class ShaplyScorer:
    """
Class to compute SHAP values for a given model and dataset.
Initialize with a model and a dataset.
Dataset must be a csv file with columns 'text' and 'label'(0,1).

## Example

```python
scorer = ShaplyScorer(input_data_path='data.csv', output_data_path='shap_values.csv')
scorer.load_model(type='hf', checkpoint='xlm-roberta-base')
scorer.load_data(subset=0.1)
scorer.init_shape_explainer()
scorer.run_shap_sequential(label_value='LABEL_0')
scorer.shap_values_for_words
scorer.save_shap_values()
```

### Methods

- `__init__(self, input_data_path:str, output_data_path:str)`
    - Initializes the `ShaplyScorer` object with the input and output data paths.
- `load_model(self, type:str, checkpoint:str)`
    - Loads a pre-trained model of the specified type and checkpoint.
- `load_data(self, subset:float=1.0)`
    - Loads the input data from the CSV file and selects a subset of rows.
- `init_shape_explainer(self)`
    - Initializes the SHAP explainer for the loaded model.
- `run_shap_sequential(self, label_value:str)`
    - Runs the SHAP explainer on each row of the dataset sequentially.
- `run_shap_multithread(self, label_value:str)`
    - Runs the SHAP explainer on each row of the dataset using multithreading.
- `save_shap_values(self)`
    - Saves the computed SHAP values to a CSV file.
"""
    
    def __init__(self, input_data_path:str, output_data_path:str):
        """
        input_data_path: path to csv file with columns 'text' and 'label'(0,1)
        output_data_path: path to csv file to save shap values
        """
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
    
    def __load_hf_model__(self, checkpoint:str):
        """
        Load a huggingface model from a checkpoint.
        """
        device=0 if torch.cuda.is_available() else -1
        model = transformers.XLMRobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
        pipe = transformers.pipeline("text-classification", model=model, tokenizer="xlm-roberta-base", device=device)
        return pipe
    
    def __load_custom_model__(self, checkpoint:str):
        """
        Load a custom model from a checkpoint.
        """
        raise NotImplementedError
    
    def load_model(self, type:str, checkpoint:str):
        """
        Load a model from a checkpoint.
        Supported types: 'hf' (huggingface), 'custom' (custom pytorch model)"""
        if type == 'hf':
            model = self.__load_hf_model__(checkpoint)
        elif type == 'custom':
            model = self.__load_custom_model__(checkpoint)
        
        self.model = {
            'model': model,
            "type" : type,
        }
        
    def load_data(self, subset:float=1.0):
        """
        load data from input_data_path and sample a subset of rows.
        The compuation of shap values is very slow, so it is recommended to sample a subset of rows.
        """
        self.data = pd.read_csv(self.input_data_path)
        self.data = self.data[['text', 'label']].dropna()
        # compute number of rows to sample
        n_rows = int(len(self.data) * subset)
        # random sample
        print(f"Sampling {n_rows} rows from {len(self.data)} rows")
        self.data = self.data.sample(n=n_rows)
        
    def init_shape_explainer(self):
        """
        Initialize a shap explainer.
        """
        masker = shap.maskers.Text(mask_token="...") # mask token is used to mask words
        model = self.model['model']
        self.explainer = shap.Explainer(model, masker)
        
    def run_shap_sequential(self, label_value:str):
        """ 
        Run shap explainer on each row of the dataset.
        """
        shap_values = []
        for text in tqdm(self.data.text, total=len(self.data), desc='Running SHAP'):
            if len(text.split()) < 2:
                continue
            
            list_of_string = [text]
            value = self.explainer(list_of_string)
            shap_values.append(value[:,:,label_value])
        return shap_values
    
    def run_shap_multithread(self, label_value:str):
        """ 
        Run shap explainer on each row of the dataset using multithreading.
        """
        raise NotImplementedError
                     
    def run_shap(self, label_value:str="LABEL_1", multithread:bool=True):
        """
        Run shap explainer on each row of the dataset.
        The shap values are computed for the label specified by label_value.
        """ 
        if label_value not in ["LABEL_1", "LABEL_2"]:
            raise ValueError("label_value must be either LABEL_1 or LABEL_2")
        if multithread:
            values = self.run_shap_multithread(label_value)
        else:
            values = self.run_shap_sequential(label_value)
        self.shap_values = values
        return values

    
    def shap_values_for_words(self):
        """
        Compute the average shap value for each word in the dataset.
        """
        self.dictionary = {}
        for value in self.shap_values:
            list_of_words = value.data[0].tolist()
            for idx, word in enumerate(list_of_words):
                # lower case
                word = word.lower()
                # remove punctuation and spaces
                word = re.sub(r'[^\w\s]+', '', word).strip()
                # if word is not in dictionary, add it
                if word not in self.dictionary:
                    self.dictionary[word] = (value.values[0][idx], 1)
                else:
                    self.dictionary[word] = (value.values[0][idx] + self.dictionary[word][0], self.dictionary[word][1] + 1)
                    
        # compute average
        for key, value in self.dictionary.items():
            self.dictionary[key] = value[0] / value[1]
            
    def save_shap_values(self):
        """
        Save shap values to output_data_path.
        """
        # create dictionary with separate keys for 'word' and 'shap_value'
        data_dict = {'word': [], 'shap_value': []}
        for key, value in self.dictionary.items():
            data_dict['word'].append(key)
            data_dict['shap_value'].append(value)
        # create DataFrame from dictionary
        df = pd.DataFrame(data_dict)
        df.to_csv(self.output_data_path, index=False)
        
    def pipeline(self,
        model_type:str,
        model_checkpoint:str,
        data_subset:float=1.0,
        label_value:str="LABEL_1",
        multi_thread:bool=False,
    ):
        print("Loading model and data, initializing SHAP")
        self.load_model(model_type, model_checkpoint)
        self.load_data(subset=data_subset)
        self.init_shape_explainer()
        print("Running SHAP")
        self.run_shap(label_value=label_value, 
                      multithread=multi_thread)
        print("Computing SHAP values for words, saving to output_data_path")
        self.shap_values_for_words()
        self.save_shap_values()
        

