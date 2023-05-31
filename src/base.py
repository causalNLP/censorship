from abc import ABC, abstractmethod
import torch
import shap
import pandas as pd
from tqdm import tqdm
import re
import logging

class Run(ABC):
    
    @abstractmethod
    def finetune(self):
        pass
    
    @abstractmethod
    def explain(self):
        pass
    
    @abstractmethod
    def search_hyperparams(self):
        pass

class Trainer(ABC):
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    @abstractmethod
    def load_data(self, sample_frac=1.0) -> tuple:
        pass
    
    @abstractmethod
    def get_data_collator(self):
        pass
    
    @abstractmethod
    def load_model(self) -> tuple:
        pass
    
    @abstractmethod
    def train(self,
              output_dir:str,
              learning_rate:float,
              batch_size:int,
              weight_decay:float,
              epochs:int):
        """
        Train model with given hyperparameters and save to output_dir

        args:
        - output_dir: directory or path to save model (depending on model type)
        - learning_rate: learning rate for optimizer
        - batch_size: batch size for training
        - weight_decay: weight decay for optimizer
        - epochs: number of epochs to train
        """
        pass
    
    # @abstractmethod
    def search_hyperparams(self,
                           output_dir:str,
                           search_space:dict= None,
                           epochs:int=3,
                           ):
        if search_space is None:
            search_space = {
                "learning_rate": [1e-6, 1e-5, 1e-4, 1e-3],
                "weight_decay": [0.0, 0.01, 0.1],
                "batch_size": [32]
            }
        best_accuracy = 0
        bets_param = {}
        for lr in search_space["learning_rate"]:
            pick_random_wd = np.random.choice(search_space["weight_decay"])
            pick_random_batch = np.random.choice(search_space["batch_size"])
            accuracy = self.train(output_dir=output_dir,
                       learning_rate=lr,
                       batch_size = pick_random_batch,
                       weight_decay=pick_random_wd,
                       epochs=epochs)
            if accuracy > best_accuracy:
                best_param = {
                    "learning_rate": lr,
                    "weight_decay": pick_random_wd,
                    "batch_size": pick_random_batch
                }
                best_accuracy = accuracy
            # save best hyperparameters
            
        return best_param, best_accuracy
                               
    
class ShapleyScorer(ABC):
    """ 
    Class to compute shapley values for a given model and dataset.
    """
    def __init__(self, input_data_path:str,
                 output_data_path:str,
                 checkpoint:str,
                 subset:float
        ):
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data =self.load_data()
        self.model = self.load_model(checkpoint)
        self.explainer = self.get_explainer()
    

    def load_data(self, subset:float=0.00001):
        """
        Load data from input_data_path and sample a subset of the data.
        """
        data = pd.read_csv(self.input_data_path)
        data = data[['text', 'label']].dropna()
        # compute number of rows to sample
        n_rows = int(len(data) * subset)
        # random sample
        logging.warning(f"Sampling {n_rows} rows from {len(data)} rows")
        data = data.sample(n=n_rows)
        return data
        
    def run_shap(self, label_value):
        """ 
        Run shap explainer on each row of the dataset.
        
        args:
        - label_value: label value to compute shap values for. It depend from the model. could be "LABEL_0" or "LABEL_1" using transformers pipeline or 0 1 using custom models. 
        """
        shap_values = []
        for text in tqdm(self.data.text, total=len(self.data), desc='Running SHAP'):
            if len(text.split()) < 2:
                continue
            
            list_of_string = [text]
            try:
                value = self.explainer(list_of_string)
            except:
                continue
    
            shap_values.append(value[:,:,label_value])
        self.shap_values = shap_values
        return shap_values
    
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
            try:
                self.dictionary[key] = value[0] / value[1]
            except:
                self.dictionary[key] = -999999
                
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
    
    @abstractmethod
    def load_model(self,checkpoint:str):
        pass
    
    @abstractmethod
    def get_explainer(self) -> shap.Explainer:
        pass
    

    