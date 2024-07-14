import os
import gc
import json
import re
import string
import torch
import numpy as np
import logging
from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel, AdamW, BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from datasets import Dataset

# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# from tensorflow import keras
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score
import argparse


def logging_storage(logfile_path):
    logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
    logging.info(torch.__version__)
    logging.info(DEVICE)


def preprocess(text, tokenizer):
    print(text)

    # convert to lowercase
    text_lower = text.lower()
    print(text_lower)

    # remove numbers
    text_no_number = re.sub(r'\d+', '', text_lower)
    print(text_no_number)

    # remove punctuation
    text_no_punc = text_no_number.translate(str.maketrans("", "", string.punctuation))
    print(text_no_punc)

    # remove whitespaces
    text_no_whitespace = text_no_punc.strip()
    print(text_no_whitespace)

    # tokenize
    text_tokenize = tokenizer.tokenize(text_no_whitespace)

    # remove stop words
    # stop_words = set(stopwords.words(‘english’))
    # result = [i for i in text_tokenize if not i in stop_words]

    return text_tokenize


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    labels_flat = labels_flat.cpu().detach().numpy()
    return np.sum(pred_flat == labels_flat), pred_flat


def train(i, t_dataloader):
    model.train()
    total_loss = 0.0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    train_len = 0
    f_acc = 0

    ## adaptive lr
    optimizer.param_groups[0]['lr'] *= (0.1) ** (1 / denom)

    for step, (b_input_ids, b_input_mask, b_labels) in enumerate(t_dataloader):
        b_input_ids = b_input_ids.to(DEVICE)
        b_input_mask = b_input_mask.to(DEVICE)
        b_labels = b_labels.to(DEVICE)
        if b_labels.size(0) <= 1:
            continue

        optimizer.zero_grad()

        torch.cuda.empty_cache()

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        pred = outputs[1].detach().cpu().numpy()
        batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
        f_acc += float(batch_f_acc)
        loss = outputs[0]
        loss.sum().backward()
        optimizer.step()

        labels_flat = b_labels.flatten().cpu().detach().numpy()
        total_actual_label = np.concatenate((total_actual_label, labels_flat))
        total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

        total_loss += float(outputs[0].sum())
        train_len += b_input_ids.size(0)

        # if step % 100 == 0 and step:
        #     precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label,
        #                                                                        total_predicted_label, average='macro')
        #     logging.info(
        #         "Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" % (
        #         train_len * 100.0 / train_inputs.size(0), i+1, step, total_loss / train_len, f_acc * 100.0 / train_len,
        #         precision * 100., recall * 100., f1_measure * 100.))

        if torch.cuda.device_count() > 1:
            p = 100
            path = save_model_path + '/e_' + str(i) + "_" + str(p) + ".ckpt"
            torch.save(model.module.state_dict(), path)
        else:
            path = save_model_path + '/e_' + str(i) + ".pt"
            torch.save(model.state_dict(), path)

    precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label, total_predicted_label,
                                                                       average='macro')
    logging.info(
        "Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" % (
        train_len * 100.0 / train_inputs.size(0), i+1, step, total_loss / train_len, f_acc * 100.0 / train_len,
        precision * 100., recall * 100., f1_measure * 100.))
    k_result['t_loss'].append(total_loss / train_len)
    k_result['t_accuracy'].append(f_acc * 100.0 / train_len)
    k_result['t_precision'].append(precision * 100.)
    k_result['t_recall'].append(recall * 100.)
    k_result['t_F1_measure'].append(f1_measure * 100.)

    if (total_loss / train_len) < early_stop:
        logging.info("Epoch: %d\tearly stopped at loss: %5.5f" % (i + 1, total_loss / train_len))

        path = save_model_path + '/early_stopped.pt'
        torch.save(model.state_dict(), path)

        flag = True
    return flag

def eva(v_dataloader):
    model.eval()
    val_len = 0
    total_loss = 0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    f_acc = 0

    with torch.no_grad():
        for step, (b_input_ids, b_input_mask, b_labels) in enumerate(v_dataloader):
            b_input_ids = b_input_ids.to(DEVICE)
            b_input_mask = b_input_mask.to(DEVICE)
            b_labels = b_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = outputs[1].detach().cpu().numpy()
            batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
            f_acc += float(batch_f_acc)

            labels_flat = b_labels.flatten().cpu().detach().numpy()
            total_actual_label = np.concatenate((total_actual_label, labels_flat))
            total_predicted_label = np.concatenate((total_predicted_label, pred_flat))

            val_len += b_input_ids.size(0)
            total_loss += float(outputs[0].sum())

        # if step % 100 == 0 and step:
        #     precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label,
        #                                                                        total_predicted_label, average='macro')
        #     logging.info(
        #         "Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" % (
        #         val_len * 100.0 / val_inputs.size(0), i+1, step, total_loss / val_len, f_acc * 100.0 / val_len,
        #         precision * 100., recall * 100., f1_measure * 100.))

        precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label, total_predicted_label,
                                                                           average='macro')
        logging.info(
            "Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" % (
            val_len * 100.0 / val_inputs.size(0), i+1, step, total_loss / val_len, f_acc * 100.0 / val_len,
            precision * 100., recall * 100., f1_measure * 100.))
        k_result['e_loss'].append(total_loss / val_len)
        k_result['e_accuracy'].append(f_acc * 100.0 / val_len)
        k_result['e_precision'].append(precision * 100.)
        k_result['e_recall'].append(recall * 100.)
        k_result['e_F1_measure'].append(f1_measure * 100.)
    return f_acc * 100.0 / val_len


# check gpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='Learning Rate', default=5, type=float)
parser.add_argument('--epochs', help='Number of Epochs', default=20, type=int)
parser.add_argument('--ml', help='Max Len of Sequence', default=1024, type=int)
parser.add_argument('--bs', help='Batch Size', default=8, type=int)
parser.add_argument('--es', help='Early Stopping Loss Criteria ', default=2, type=float)
parser.add_argument('--adaptive', help='Adaptive LR', default='20', type=float)

args = parser.parse_args()

lr = 10 ** (-args.lr)
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
early_stop = 10 ** (-args.es)
# test_size = args.ts
model_str = 'xlnet'
# model_str = 'bert'
num_labels = 4
denom = args.adaptive

# set path
trg_path = "moody_lyrics.json"
ending_path = ('%s_%d_bs_%d_adamw_lr_%s_es_%s_%d' %(model_str, MAX_LEN, batch_size, str(lr).replace("-",""), str(early_stop).replace("-",""), denom))
save_model_path = "models/" + ending_path
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
if not os.path.exists("logs/"):
    os.mkdir("logs/")
logfile_path = "logs/" + ending_path
logging_storage(logfile_path)
# result_path = "result_json/" + ending_path
if not os.path.exists("result_json/"):
    os.makedirs("result_json/")


# fetch data
with open(trg_path) as f:
    song_info = json.load(f)
lyrics = song_info["Lyric"][0]
labels = song_info["Mood"][0]

# convert categorical labels to numerical labels
# Angry = 0; Happy = 1; Relaxed = 2; Sad = 3
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

attention_masks = []
# input_ids = []
# tokenize
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

# for i, lyric in enumerate(lyrics):
#     encodings = tokenizer.encode_plus(lyrics, add_special_tokens=True, max_length=MAX_LEN, return_tensors='pt',
#                                       return_token_type_ids=False, return_attention_mask=True, pad_to_max_length=True)
#     attention_mask = pad_sequences(encodings['attention_mask'], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
#     attention_masks.append(attention_mask)
#     input_id = pad_sequences(encodings['input_ids'], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
#     input_ids.append(input_id)

tokenized_texts = [tokenizer.tokenize(lyric) for lyric in lyrics]
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# for lyric in lyrics:
#     lyric = preprocess(lyric, tokenizer)
#     print(lyric)

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# train-validation test split
# 6 2 2
train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(input_ids, encoded_labels,
                                                                                random_state=SEED, test_size=0.2)
train_val_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                     random_state=SEED, test_size=0.2)

# Convert to Tensor
train_val_inputs = torch.tensor(train_val_inputs)
train_val_labels = torch.tensor(train_val_labels)
train_val_masks = torch.tensor(train_val_masks)

test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_masks)

# dataloader
# train_val_data = TensorDataset(train_val_inputs, train_val_masks, train_val_labels)
# train_val_sampler = RandomSampler(train_val_data)
# train_val_dataloader = DataLoader(train_val_data, sampler=train_val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

k_folds = 5
results = []
result_json = {}
kfold = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_inputs)):
    logging.info('------------------------------------------------------')
    logging.info('%d FOLD', fold+1)

    # write results of each fold into a dic
    k_result = {'t_loss': [], 't_accuracy': [], 't_precision': [], 't_recall': [], 't_F1_measure': [],
                'e_loss': [], 'e_accuracy': [], 'e_precision': [], 'e_recall': [], 'e_F1_measure': [],
                'epoch': []}

    # Dataset.select(indices=train_idx)
    train_inputs, val_inputs = train_val_inputs[train_idx], train_val_inputs[val_idx]
    train_masks, val_masks = train_val_masks[train_idx], train_val_masks[val_idx]
    train_labels, val_labels = train_val_labels[train_idx], train_val_labels[val_idx]

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # define model
    # xlnet
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)
    # BERT
    # model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
    # model = nn.DataParallel(model)
    model.to(DEVICE)

    # define optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    es_flag = False

    for i in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        es_flag = train(i, train_dataloader)

        if es_flag:
            break

    k_result['epoch'].append(i+1)
    gc.collect()
    torch.cuda.empty_cache()
    k_acc = eva(val_dataloader)
    results.append(k_acc)
    result_json[str(fold+1)] = []
    result_json[str(fold+1)].append(k_result)

logging.info("AVERAGE ACCURACY: %5.3f", sum(results) / len(results))
result_json['average accuracy'] = []
result_json['average accuracy'].append(sum(results) / len(results))
result_path = "result_json/" + ending_path + '.json'
with open(result_path, 'w') as f:
    json.dump(result_json, f, indent=4)
