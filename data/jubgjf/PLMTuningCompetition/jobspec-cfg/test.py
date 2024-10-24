"""
Given trained prompt and corresponding template, we ask test API for predictions
Shallow version
We train one prompt after embedding layer, so we use sentence_fn and embedding_and_attention_mask_fn.
Baseline code is in bbt.py
"""
seed = 8

import os
import torch
from test_api import test_api
from test_api import RobertaEmbeddings
from transformers import RobertaConfig, RobertaTokenizer
# from models.modeling_roberta import RobertaModel
import numpy as np
import csv
import random

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


def sentence_fn_factory(task_name):
    prompt_initialization = tokenizer.decode(list(range(1000, 1050)))
    if task_name in ['QQP', 'QNLI', 'SNLI']:
        def sentence_fn(test_data):
            return "%s . %s ? %s , %s" % (prompt_initialization, test_data, tokenizer.mask_token, test_data)
            # return prompt_initialization + ' . ' + test_data + ' ? <mask> , ' + test_data
    elif task_name in ['SST-2']:
        def sentence_fn(test_data):
            return prompt_initialization + ' . ' + test_data + ' . It was <mask> .'
    elif task_name in ['DBPedia']:
        def sentence_fn(test_data):
            return '%s . %s News: %s' % (prompt_initialization, tokenizer.mask_token, test_data)
    else:
        raise ValueError

    return sentence_fn


# %s . %s . It was %s .


# %s . %s News: %s
verbalizer_dict = {
    'SNLI': ["Yes", "Maybe", "No"],
    'SST-2': ["bad", "great"],
    'DBPedia': [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork"
    ],
    'QNLI': ["No", "Yes"],
    'QQP': ["No", "Yes"]
}

device = 'cpu'

for task_name in ['QQP','SST-2','DBPedia']:
#for task_name in verbalizer_dict.keys():
    for seed in [8]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        best = torch.load(f'./results/{task_name}/{seed}/best.pt', map_location='cpu').to(device).view(50, -1)

        sentence_fn = sentence_fn_factory(task_name)


        def embedding_and_attention_mask_fn(embedding, attention_mask):
            # res = torch.cat([init_prompt[:-5, :], input_embed, init_prompt[-5:, :]], dim=0)
            prepad = torch.zeros(size=(1, 1024), device=device)
            pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
            return embedding + torch.cat([prepad, best, pospad]), attention_mask


        predictions = torch.tensor([], device=device)
        for res, _, _ in test_api(
                sentence_fn=sentence_fn,
                embedding_and_attention_mask_fn=embedding_and_attention_mask_fn,
                test_data_path=f'./test_datasets/{task_name}/encrypted.pth',
                task_name=task_name,
                device=device
        ):
            verbalizers = verbalizer_dict[task_name]
            intrested_logits = [res[:, tokenizer.encode(verbalizer, add_special_tokens=False)[0]] for verbalizer in
                                verbalizers]

            pred = torch.stack(intrested_logits).argmax(dim=0)
            predictions = torch.cat([predictions, pred])

        if not os.path.exists(f'./predictions/{task_name}'):
            os.makedirs(f'./predictions/{task_name}')
        with open(f'./predictions/{task_name}/{seed}.csv', 'w+') as f:
            wt = csv.writer(f)
            wt.writerow(['', 'pred'])
            wt.writerows(
                torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())
