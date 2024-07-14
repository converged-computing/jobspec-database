import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device is: " + device)

sentences = pickle.load(open( "./data/sentences.pkl", "rb" ))

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

encoded_input = tokenizer(sentences[:].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
model = model.to(device)

# compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# mean pooling
embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
embeddings = embeddings.to('cpu')
pickle.dump(embeddings, open( "./data/embeddings.pkl", "wb" ))

