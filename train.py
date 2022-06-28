import json
import torch
from models import *
from utils import *

EPOCHS = 4
BATCH_SIZE = 32
ACCUMULATION_STEPS = 5
LEARNING_RATE = 1e-5

config_path = './config/electra_1.json'
single_model_config = json.load(open(config_path, 'r'))

seed = 42
seed_everything(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if single_model_config['model_type'] == 'BERT':
    print("===Use BERT model===")
    checkpoint_dir = '/content/drive/MyDrive/VLSP-Fake-News-Detection/trained_models/vbert_base/'
    tokenizer = BertTokenizer.from_pretrained(
      single_model_config['model_name'], do_lower_case=False)
    tokenizer.add_tokens(['<url>'])
    config = BertConfig.from_pretrained(single_model_config['model_name'], num_labels=2,
                                      output_hidden_states=True)
    model = BertReINTELClassification.from_pretrained(
      single_model_config['model_name'], config=config)
    model.to(device)
    tsfm = model.bert
elif single_model_config['model_type'] == 'ROBERTA':
    print("===Use PhoBERT model===")
    checkpoint_dir = '/content/drive/MyDrive/VLSP-Fake-News-Detection/trained_models/phobert_base/'
    tokenizer = AutoTokenizer.from_pretrained(
        single_model_config['model_name'])
    tokenizer.add_tokens(['<url>'])
    config = RobertaConfig.from_pretrained(single_model_config['model_name'], num_labels=2,
                                            output_hidden_states=True)
    model = PhoBERTReINTELClassification.from_pretrained(
        single_model_config['model_name'], config=config)
    # model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    tsfm = model.roberta
elif single_model_config['model_type'] == 'ELECTRA':
    print("===Use ELECTRA model===")
    checkpoint_dir = '/content/drive/MyDrive/VLSP-Fake-News-Detection/trained_models/electra/'
    tokenizer = ElectraTokenizer.from_pretrained(
        single_model_config['model_name'], do_lower_case=False)
    tokenizer.add_tokens(['<url>'])
    config = ElectraConfig.from_pretrained(single_model_config['model_name'], num_labels=2,
                                            output_hidden_states=True, output_attentions=False)
    model = ElectraReINTELClassification.from_pretrained(
        single_model_config['model_name'], config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    tsfm = model.electra
else:
  print("Model type invalid!!!")

print(f"Seed number: {seed}")