import pandas as pd
from src.embeddings import DatetimeEmbedding, RobertaTokenizer

from torch.utils.data import Dataset, DataLoader
from torchtext.functional import to_tensor
from torchtext.models import XLMR_BASE_ENCODER

import torch

class CensorDataset(Dataset):
    def __init__(self, path = "data/dataset_train.csv", lang = None):
        self.date_embedder = DatetimeEmbedding()
        self.tokenizer = XLMR_BASE_ENCODER.transform()
        dataframe = pd.read_csv(path)
        
        self.dataframe = self.preprocess(dataframe)

        self.max_len = 512
        if lang is not None:
            self.filter_lang(lang)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["text_proc"]
        censored = row["censored"]
        
        # if text < 512 tokens, pad it
        
        # tokenize
        tokens = to_tensor(self.tokenizer(text))
        # pad
        tokens = self.pad(tokens)
        
        return tokens, censored
        
    def preprocess(self, df):
        df = df[["lang", "created_at", "text_proc", "censored"]]
        # remove nan
        df = df.dropna()
        return df
    
    def pad(self, tokens):
        pad_len = self.max_len - len(tokens)
        pad = torch.ones(pad_len, dtype=torch.int64)
        return torch.cat([tokens, pad])
    
    def filter_lang(self, lang):
        # check if the lang is in the dataset
        try:
            langs = self.dataframe["lang"].unique()
            assert lang in langs
        except:
            print(f"Language {lang} not in dataset or misspelled")
            
        self.dataframe = self.dataframe[self.dataframe["lang"] == lang]

    def get_min_max_date(self):
        min_date = self.dataframe["created_at"].min()
        max_date = self.dataframe["created_at"].max()
        return min_date, max_date

    def get_timeline(self, path):
        emb_model = XLMR_BASE_ENCODER.get_model()
        timeline = pd.read_csv(path)
        timeline = timeline[["Date", "Description"]]
        date_embedded = []
        for date in timeline["Date"]:
            date_embedded.append(self.date_embedder(date, dateformat = '%m/%d/%y'))
        

        # tokenize the description
        desc = []
        for description in timeline["Description"]:
            description_embedded = self.tokenizer(description)
            # to tensor
            description_embedded = to_tensor(description_embedded)
            # pad the description
            description_embedded = self.pad(description_embedded)
            desc.append(description_embedded)
        
        # from list of tensors to tensor
        description_embedded = torch.stack(desc)
        # get the embedding of the description
        description_embedded = emb_model(description_embedded)
        
        # get cls for each description
        description_embedded = description_embedded[:, 0, :]
    
        # for each desc in timeline, stack the date embedding
        date_embedded = torch.stack(date_embedded)
        total_embedded = torch.cat([description_embedded, date_embedded], dim=1)
        
        return total_embedded

    def get_weights(self):
        # get the number of censored and uncensored tweets
        censored = self.dataframe[self.dataframe["censored"] == 1]
        uncensored = self.dataframe[self.dataframe["censored"] == 0]
        # get the number of censored and uncensored tweets
        n_censored = len(censored)
        n_uncensored = len(uncensored)
        # get the total number of tweets
        n_total = n_censored + n_uncensored
        # get the weights
        w_censored = n_total / (3*n_censored)
        w_uncensored = n_total / (3*n_uncensored)
        
        weights = torch.tensor([w_uncensored, w_censored])
        return weights