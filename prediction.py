import torch
import pandas as pd
import torch.nn.functional as F

from torch import cuda
from transformers import *
from models import *

def electra_prediction(text, model, tokenizer, device, max_len = 256):
    inputs = tokenizer(text, padding = "max_length", max_length = max_len,
                       truncation = True, return_tensors = "pt")
    model.to(device)
    input_ids = inputs["input_ids"].squeeze(1).to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    outputs = model(input_ids, attention_mask)
    outputs = torch.sigmoid(outputs)[0].detach().cpu().numpy()
    # print(outputs[0].detach().cpu().numpy())
    if outputs[:] > 0.5:
        prediction = "fake"
    else:
        prediction = "real"
    # prediction = outputs.argmax(dim = 1).cpu().numpy()
    return prediction

device = 'cuda' if cuda.is_available() else 'cpu'

dataframe = pd.read_csv("./final_private_test.csv")
sample_txt = dataframe['post_message'][3]

tokenizer = ElectraTokenizer.from_pretrained("./train models/electra/", do_lower_case=False)
tokenizer.add_tokens(["<url>"])

config_path = "./train models/electra/model_7370.bin"
config = ElectraConfig.from_pretrained("./train models/electra/", num_labels = 1, output_hidden_states = True)

model = ElectraReINTELClassification.from_pretrained(config_path, config=config)

print(electra_prediction(sample_txt, model, tokenizer, device))