import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchtext
from torchtext.models import XLMR_BASE_ENCODER, RobertaClassificationHead
from torchtext.functional import to_tensor



import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

import numpy as np
import os
import re
import logging, sys

from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Union, Iterable, Callable





class Config:
    hidden_size:int = 768
    date_embedding_size:int = 5
    query_size:int = 64
    num_event:int = 9
    
    def asdict(self):
        conf_dict = {
            "hidden_size": self.hidden_size,
            "date_embedding_size": self.date_embedding_size,
            "query_size": self.query_size
        }
        return conf_dict
    


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        ## roberta base
        self.xlmr_roberta_base = torchtext.models.XLMR_BASE_ENCODER.get_model(freeze_encoder = False)
        self.classification_layer = nn.Sequential(
            nn.Linear(Config.hidden_size, Config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(Config.hidden_size, 2),
        )
        ## date embedding

    def forward(self, tweet_tokens): ## [batch_size, max_seq_len], [batch_size, max_seq_len]
        #apply the attention mask
        
        # get the tweet embedding
        sentence_embedding = self.xlmr_roberta_base(tweet_tokens) # [batch_size, max_seq_len, hidden_size]
        sentence_embedding = sentence_embedding[:, 0, :] # [batch_size, hidden_size]
        
        output = self.classification_layer(sentence_embedding) # [batch_size, 2]
        
        #get the logits
        output = F.log_softmax(output, dim=1) # [batch_size, 2]
  
        return output