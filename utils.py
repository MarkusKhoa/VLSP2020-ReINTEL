from vncorenlp import VnCoreNLP
from nltk.tokenize import TweetTokenizer
from pandas import DataFrame

import re
import torch
import json
import os
import numpy as np

def seed_everything(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def save_checkpoint(model, tokenizer, checkpoint_path, epoch='best'):
  torch.save(model.state_dict(), os.path.join(
      checkpoint_path, f'model_{epoch}.bin'))
  model.config.to_json_file(os.path.join(checkpoint_path, 'config.json'))
  tokenizer.save_vocabulary(checkpoint_path)
  
def convert_tokens_to_ids(text, tokenizer, max_len = 256):
  inputs = tokenizer.encode_plus(text, padding = "max_length", max_length = max_len, truncation = True)
  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  
  return torch.tensor(input_ids, dtype = torch.long), torch.tensor(attention_mask, dtype = torch.long)