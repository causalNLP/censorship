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

from datetime import datetime


class DatetimeEmbedding():
    def __init__(self, dateformat = '%Y-%m-%d %H:%M:%S'):
        month, day, seconds = self.create_dict_map()
        self.month_dict = month
        self.day_dict = day
        self.seconds_dict = seconds
        self.dateformat = dateformat
        
    def __call__(self, date, dateformat = '%Y-%m-%d %H:%M:%S'):
        if isinstance(date, list):
            return self.datetime_embedding_batch(date, dateformat)
        return self.datetime_embedding(date, dateformat)
        
    def create_dict_map(self):
        try :
            from feature_engine.creation import CyclicalFeatures
        except:
            os.system('pip install feature_engine')
            from feature_engine.creation import CyclicalFeatures
        #create a vector of month
        month = np.array([i for i in range(1,13)]).reshape(-1,1)
        day = np.array([i for i in range(1,32)]).reshape(-1,1)
        second_in_day = np.array([i for i in range(0,86400)]).reshape(-1,1)
        cf = CyclicalFeatures()
        month = cf.fit_transform(month)
        day = cf.fit_transform(day)
        second_in_day = cf.fit_transform(second_in_day)
        
        #create a dictionary
        month_dict = {}
        for i in range(0,12):
            month_dict[i+1] = [month["x0_cos"][i], month["x0_sin"][i]]
        # create a dictionary
        day_dict = {}
        for i in range(0,31):
            day_dict[i+1] = [day["x0_cos"][i], day["x0_sin"][i]]
        
        # create a dictionary
        second_dict = {}
        for i in range(0,86400):
            second_dict[i] = [second_in_day["x0_cos"][i], second_in_day["x0_sin"][i]]
        
        
        return month_dict, day_dict, second_dict   
    
    def datetime_embedding(self,date:str, dateformat = '%Y-%m-%d %H:%M:%S'):
        if dateformat == '%Y-%m-%d %H:%M:%S':
            date = self.parse_date(date)
        date = datetime.strptime(date, dateformat)
        seconds = self.convert_h_to_s(date.hour, date.minute, date.second)
        
        month = self.month_dict[date.month]
        day = self.day_dict[date.day]
        # second = self.seconds_dict[seconds]
        
        # combine the two lists
        month_day_year = month + day + [date.year]
        
        # to tensor
        month_day_year = torch.tensor(month_day_year, dtype=torch.float32)
        # expand for the batch
        #month_day_year = month_day_year.unsqueeze(0)
    
        return month_day_year
    
    def parse_date(self, date:str):
        pattern = r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.\d{3}Z$"
        match = re.match(pattern, date)

        if match:
            year = match.group(1)
            month = match.group(2)
            day = match.group(3)
            hour = match.group(4)
            minute = match.group(5)
            second = match.group(6)

            formatted_date = f"{year}-{month}-{day} {hour}:{minute}:{second}"
            return formatted_date
        else:
            print("Invalid date format.")

    def convert_h_to_s(self, h, m, s):
        return h*3600 + m*60 + s
        

class TimelineEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlmr_roberta_base = torchtext.models.XLMR_BASE_ENCODER.get_model()
    
    def forward(self, timeline_tokens, datetime_embedding): # [num_event, max_len], [num_event, date_embedding_size]
        event_embeddings = self.xlmr_roberta_base(timeline_tokens)
        event_embedding_cls = event_embeddings[:, 0, :]
        output = torch.cat([event_embedding_cls, datetime_embedding], dim=1)
        return output
    
class RobertaTokenizer():
    def __init__(self):
        self.tokenizer = torchtext.models.XLMR_BASE_ENCODER.transform()
    
    def __call__(self, list_text:list):
        return to_tensor(self.tokenizer(list_text), padding_value=1)