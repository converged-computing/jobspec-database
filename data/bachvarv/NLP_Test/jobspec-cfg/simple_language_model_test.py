import os
import random
import subprocess
import time

import transformers
import tensorflow as tf
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import csv

from simple_language.seq2seq_data.SimpleLanguageModel import SimpleLanguageModel
from simple_language.transformer_data.MaskedLoss import MaskedLoss


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def save_history(history, file):

    with open(file, 'a+') as f:
        for i in range(len(history.history['accuracy'])):
            line = f"{history.history['accuracy'][i]}\t{history.history['loss'][i]}\n"
            f.writelines(line)
        f.close()

model_name = os.path.join(os.curdir, 'bert-base-german-cased')
path_to_history = os.path.join(os.curdir, 'history_nmt_mix_v1.csv')

if not os.path.isdir(model_name):
    url = 'https://huggingface.co/bert-base-german-cased'
    git("clone", url)



# The Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens(['[GO]'])
vocab_size = tokenizer.vocab_size
print(vocab_size)

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9))
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)


# Config variables
cfg = dict(
    max_sentence=128,
    hidden_layer_size=768,
    batch_size=1,
    transformer_heads=12,
    head_size=64
)

# Corpus preparations
input_ids = []
token_type_ids = []
attention_masks = []
labels = []

eval_ids = []
eval_type_ids = []
eval_masks = []
eval_labels = []

input_arr = []
eval_arr = []
label_arr = []
eval_label_arr = []
path_to_corpus = os.path.join(os.path.join(os.curdir, 'corpus_for_test'), 'einfache_sprache.csv')
path_to_corpus_2 = os.path.join(os.path.join(os.curdir, 'corpus_for_test'), 'spd_programm_einfache_sprache_v1.csv')
path_to_corpus_3 = os.path.join(os.path.join(os.curdir, 'corpus_for_test'), 'simple_language_openAI.csv')
longest_cand = 0
with open(path_to_corpus, 'r', encoding='utf-8') as file:

    lines = file.readlines()
    # print(len(lines))
    for line in lines:

        x, y = line.split(sep='\t')
        # tokenized_x = tokenizer.tokenize(x)
        # tokenized_y = tokenizer.tokenize(y)
        # token_x = tokenizer(x, return_tensors='tf')
        # token_y = tokenizer(y, return_tensors='tf')
        # input_arr.append(token_x)
        # label_arr.append(token_y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10 or x == 'Anrede':
            input_arr.append(x)
            # label_arr.append('[CLS]' + y) # nmt_model_v2
            # label_arr.append('[GO]' + y) # nmt_model_v4
            label_arr.append(y) # nmt_model_masked_loss_v1
            # label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            # eval_label_arr.append('[CLS]' + y) # nmt_model_v2
            # eval_label_arr.append('[GO]' + y) # nmt_model_v4
            eval_label_arr.append( y) # nmt_model_masked_loss_v1
            # eval_label_arr.append(y)
            # eval_arr.append(tokenized_x)
            # eval_label_arr.append(tokenized_y)

        # size_x = token_x['input_ids'].shape[1]
        # size_y = token_y['input_ids'].shape[1]
        # if size_x > size_y:
        #     if size_x > longest_cand:
        #         longest_cand = size_x
        # else:
        #     if size_y > longest_cand:
        #         longest_cand = size_y


with open(path_to_corpus_2, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        x, y = line.split(sep='\t')
        # tokenized_x = tokenizer.tokenize(x)
        # tokenized_y = tokenizer.tokenize(y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
            input_arr.append(x)
            # label_arr.append('[CLS]' + y) # nmt_model_v2
            # label_arr.append('[GO]' + y) # nmt_model_v4
            label_arr.append(y) # nmt_model_masked_loss_v1
            # label_arr.append(y)
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)
            # eval_label_arr.append('[CLS]' + y) # nmt_model_v2
            # eval_label_arr.append('[GO]' + y) # nmt_model_v4
            eval_label_arr.append(y) # nmt_model_masked_loss_v1
            # eval_label_arr.append(y)
            # eval_arr.append(tokenized_x)
            # eval_label_arr.append(tokenized_y)

with open(path_to_corpus_3, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        x, y = line.split(sep='\t')
        # tokenized_x = tokenizer.tokenize(x)
        # tokenized_y = tokenizer.tokenize(y)
    # if 12 < len(tokenized_y) <= 30:
    #     print(len(tokenized_y))
        if random.randint(0, 100) > 10:
            input_arr.append(x)
            # label_arr.append(y)
            # label_arr.append('[CLS]' + y) # nmt_model_v2
            # label_arr.append('[GO]' + y) # nmt_model_v4
            label_arr.append(y) # nmt_model_masked_loss_v1
            # input_arr.append(tokenized_x)
            # label_arr.append(tokenized_y)

        else:
            eval_arr.append(x)

            # eval_label_arr.append('[CLS]' + y) # nmt_model_v2
            # eval_label_arr.append('[GO]' + y) # nmt_model_v2
            eval_label_arr.append(y) # nmt_model_masked_loss_v1
            # eval_label_arr.append(y)

# creating the dataset

#
# def tokenize_function(examples):
#     return tokenizer(examples, max_length=128, return_tensors='tf')


# arr = np.reshape(input_arr, newshape=[1, len(input_arr)])
arr_inp = tokenizer(input_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# arr_lab = tokenizer(label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
# print(arr_inp)

arr_eval = tokenizer(eval_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')
arr_eval_label = tokenizer(eval_label_arr, max_length=cfg['max_sentence'], padding='max_length', truncation=True, return_tensors='tf')

dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_inp['input_ids'],
         token_type_ids=arr_inp['token_type_ids'],
         attention_mask=arr_inp['attention_mask']),
    label_arr)).batch(1)

eval_dataset = tf.data.Dataset.from_tensor_slices((
    dict(input_ids=arr_eval['input_ids'],
         token_type_ids=arr_eval['token_type_ids'],
         attention_mask=arr_eval['attention_mask']),
    eval_label_arr)).batch(1)

# Model
model = SimpleLanguageModel(cfg['hidden_layer_size'], vocab_size, tokenizer)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3
                                                ),
              loss=MaskedLoss())

# Checkpoint
path_to_checkpoint = os.path.join(os.curdir, 'SLM_HPC_v1_without_BERT_EP20_lr1e-3')
# SLM_v4 lr=1e-4
# SLM_v5 lr=1e-3 changed loss to be calculated from logits
# SLM_v6_with_VGA lr=1e-3 to run with gpu
# SLM_v1_without_BERT
# path_to_saved_model = os.path.join(os.curdir, 'saved_model_gru_1024_v3')

ckpt = tf.train.Checkpoint(model)
ckpt_manager = tf.train.CheckpointManager(ckpt, path_to_checkpoint, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
else:
    print('Initializing from scratch!')

result, _, text = model((['Geburtsort'], ['[CLS]']))
#prediction = tf.argmax(result.logits, axis=-1)
target = tokenizer(['In welchem Land sind Sie geboren?'])['input_ids']
print(target)
#print(prediction.numpy())
# print(result.logits)
#s = tokenizer.decode(prediction[0])
print(text)
model.summary()

start = time.time()
losses = []
for i in range(1, 21):
    for inp, tar in dataset:

        logs = model.train_step((inp, tar))
        print(logs)
        losses.append(logs['batch_loss'].numpy())
    print(f'Step {i}')
end = time.time()
print()
plt.plot(losses)
plt.savefig('slm_1e-3_20EP_PT3.png')


print(end - start)

result, _, text = model((['Anrede'], ['[CLS]']))
target = tokenizer(['Herr oder Frau.'])['input_ids']
print(target)
print(text)
# print(result.logits)
""" print(prediction)
s = tokenizer.decode(prediction[0])
print(s) """

# BLEU_SCORE
pred_arr = []
for cand in eval_arr:
    print(f'Candidate: {cand}')
    _, _, text = model(([cand], ['[CLS]']))
    pred_arr.append(text)
    print(f'Prediction: {text}')
    print('_____________________________________')
bleu = corpus_bleu(eval_label_arr, pred_arr)
print(bleu)

with open('slm_bleu_table_eval_arr.csv', 'w+') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([path_to_checkpoint, str(bleu)])


ckpt_manager.save()
# for inp, tar in dataset:
#     # print(inp, tar)
#     model.train_step((inp, tar))
#     break
