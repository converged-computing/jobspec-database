import argparse
import gzip
import numpy as np
import tqdm
import json
import time
import datetime
import os
import random

import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter


from transformers.optimization import get_constant_schedule_with_warmup
from model.optimizer import get_optimizer

from torch.utils.data import DataLoader

from utils import TextSamplerDatasetS2S, MyCollateS2S, ids_to_tokens, BPE_to_eval, epoch_time, count_parameters, remove_eos

from model.transformer import Transformer

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import sacrebleu


def load_vocabulary(dataset_option):
    if dataset_option == 1:
        vocab_path = 'dataset/nl/seq2seq/en2de/wmt17_en_de/vocabulary.json'
    elif dataset_option == 2:
        vocab_path = 'dataset/nl/seq2seq/en2fr/vocabulary.json'
    else:
        raise ValueError("Invalid dataset option. Choose 1 for dataset/nl/seq2seq/en2de/wmt17_en_de or 2 for dataset/nl/seq2seq/en2fr/wmt14_en_fr.")

    with open(vocab_path, 'r') as f:
        vocabulary = json.load(f)

    return vocabulary

def train(dataset_option, finetuning):

    print(torch.cuda.device_count())
    

    ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_2 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1, ddp_kwargs_2])

    
    vocabulary = load_vocabulary(dataset_option)
    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    EPOCHS = 100
    BATCH_SIZE = 100
    LEARNING_RATE = 1e-3
    GENERATE_EVERY  = 5
    MAX_LEN = 80
    WARMUP_STEP = 0

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

   

    # Step 2: Prepare the model (original transformer) and push to GPU
    model = Transformer(
        model_dimension=512,
        src_vocab_size=NUM_TOKENS,
        trg_vocab_size=NUM_TOKENS,
        number_of_heads=8,
        number_of_layers=6,
        dropout_probability=0.1
    )

    # Step 3: Prepare other training related utilities
    ca = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # optimizer
    optimizer = get_optimizer(model.parameters(), LEARNING_RATE, wd=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEP)

    print('number of parameters:', count_parameters(model))

    if dataset_option == 1:
        train_data_src_path = 'dataset/nl/seq2seq/en2de/wmt17_en_de/train.en.ids.gz'
        train_data_tgt_path = 'dataset/nl/seq2seq/en2de/wmt17_en_de/train.de.ids.gz'
        valid_data_src_path = 'dataset/nl/seq2seq/en2de/wmt17_en_de/valid.en.ids.gz'
        valid_data_tgt_path = 'dataset/nl/seq2seq/en2de/wmt17_en_de/valid.de.ids.gz'

    

        folder_path = './output/transformer/en2de/' + str(SEED) + str('/')

    elif dataset_option == 2:
        train_data_src_path = 'dataset/nl/seq2seq/en2fr/wmt14_en_fr/train.en.ids.gz'
        train_data_tgt_path = 'dataset/nl/seq2seq/en2fr/wmt14_en_fr/train.fr.ids.gz'
        valid_data_src_path = 'dataset/nl/seq2seq/en2fr/wmt14_en_fr/valid.en.ids.gz'
        valid_data_tgt_path = 'dataset/nl/seq2seq/en2fr/wmt14_en_fr/valid.fr.ids.gz'

        folder_path = './output/transformer/en2fr/' + str(SEED) + str('/')
    else:
        raise ValueError("Invalid dataset option. Choose 1 for wmt17_en_de or 2 for wmt14_en_fr.")
        
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    # Create SummaryWriter to save  TensorBoard logs
    tensorboard_writer = SummaryWriter(log_dir=folder_path + 'logs/')

    with gzip.open(train_data_src_path, 'r') as file:
        X_train = file.read()
        X_train = X_train.decode(encoding='utf-8')
        X_train = X_train.split('\n')
        X_train = [np.array([int(x) for x in line.split()]) for line in X_train]
    

    with gzip.open(train_data_tgt_path, 'r') as file:
        Y_train = file.read()
        Y_train = Y_train.decode(encoding='utf-8')
        Y_train = Y_train.split('\n')
        Y_train = [np.array([int(x) for x in line.split()]) for line in Y_train if line != '']

    with gzip.open(valid_data_src_path, 'r') as file:
        X_dev = file.read()
        X_dev = X_dev.decode(encoding='utf-8')
        X_dev = X_dev.split('\n')
        X_dev = [np.array([int(x) for x in line.split()]) for line in X_dev if line != '']

    with gzip.open(valid_data_tgt_path, 'r') as file:
        Y_dev = file.read()
        Y_dev = Y_dev.decode(encoding='utf-8')
        Y_dev = Y_dev.split('\n')
        Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev]


    train_dataset = TextSamplerDatasetS2S(X_train, Y_train, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=True,
                           pin_memory=True, collate_fn=MyCollateS2S(pad_idx=0))
    dev_dataset = TextSamplerDatasetS2S(X_dev, Y_dev, MAX_LEN)
    dev_loader  = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=8, collate_fn=MyCollateS2S(pad_idx=0))

    
    # ...

    # with gzip.open(valid_data_src_path, 'r') as src_file, gzip.open(valid_data_tgt_path, 'r') as tgt_file:
    #     X_dev_src = src_file.read()
    #     X_dev_src = X_dev_src.decode(encoding='utf-8')
    #     X_dev_src = X_dev_src.split('\n')
    #     X_dev_src = [np.array([int(x) for x in line.split()]) for line in X_dev_src if line != '']
    #     # X_dev_src = X_dev_src[0:10]

    #     Y_dev_tgt = tgt_file.read()
    #     Y_dev_tgt = Y_dev_tgt.decode(encoding='utf-8')
    #     Y_dev_tgt = Y_dev_tgt.split('\n')
    #     Y_dev_tgt = [np.array([int(x) for x in line.split()]) for line in Y_dev_tgt if line != '']

        # Y_dev_tgt = Y_dev_tgt[0:10]

    # train_dataset = TextSamplerDatasetS2S(X_dev_src, Y_dev_tgt, MAX_LEN)
    # train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True,
    #                         pin_memory=True, collate_fn=MyCollateS2S(pad_idx=0))
    # dev_dataset = TextSamplerDatasetS2S(X_dev_src, Y_dev_tgt, MAX_LEN)
    # dev_loader  = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, collate_fn=MyCollateS2S(pad_idx=0))

    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, dev_loader, scheduler)

    if finetuning:
        print('finetune')
        model.load_state_dict(
            torch.load(
                folder_path + 'model.pt'
            ),
        )

    best_bleu = 0
    breakaction = False
    # training
    for i in tqdm.tqdm(range(EPOCHS), desc='training'):
        start_time = time.time()
        model.train()
        if breakaction:
            break

        countdown = 0
        count_loss = 0

        for src_train, tgt_train in train_loader:

            src_mask = src_train != 0
            src_mask = src_mask[:, None, None, :]

            inp_tgt, out_tgt = remove_eos(tgt_train), tgt_train[:, 1:]
            # print('inp_tgt', inp_tgt)
            # print('out_tgt', out_tgt)

            tgt_mask = model.get_masks_and_count_tokens_trg(inp_tgt)

            countdown += 1

            predicted_log_distributions = model(src_train, inp_tgt, src_mask, tgt_mask)

            # print('predicted_log_distributions', predicted_log_distributions)

            loss = ca(predicted_log_distributions.view(-1, NUM_TOKENS), out_tgt.contiguous().view(-1).type(torch.LongTensor).cuda())

            count_loss += loss.item()

            # print(loss.item())

            # if torch.isnan(loss).item():
            #     breakaction = True
            #     break

            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print('loss =', count_loss)

        # torch.save(model.state_dict(),
        #            'output/model_seq2seq_each_epoch.pt'
        #            )

        if i != 0 and i % GENERATE_EVERY == 0:

            model.eval()
            target = []
            predicted = []
            for src_dev, tgt_dev in dev_loader:

                src_mask = src_dev != 0
                src_mask = src_mask[:, None, None, :]

                sample = model.generate_greedy(src_dev, src_mask, MAX_LEN)

                # sample = accelerator.gather(sample)
                # tgt_dev = accelerator.gather(tgt_dev)

                target.append([ids_to_tokens(tgt_dev.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])
                predicted.append([ids_to_tokens(sample.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])

            target_bleu = [BPE_to_eval(sentence, lm=False) for sentence in target[0]]
            predicted_bleu = [BPE_to_eval(sentence, lm=False) for sentence in predicted[0]]

            print('target_bleu', target_bleu)
            print('predicted_bleu', predicted_bleu)

            bleu = sacrebleu.corpus_bleu(predicted_bleu, [target_bleu])

            bleu = bleu.score

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print('Epoch: {0} | Time: {1}m {2}s, bleu score = {3}'.format(i, epoch_mins, epoch_secs, bleu))

            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(model.state_dict(),
                           folder_path + 'model.pt'
                           )

                torch.save(optimizer.state_dict(), folder_path + 'optimizer.bin')

            # tensorboard save metrics
            tensorboard_writer.add_scalar('Train/Loss', count_loss, i)

            if i != 0 and i % GENERATE_EVERY == 0:
                tensorboard_writer.add_scalar('Validation/BLEU', bleu, i)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Model Training')
    parser.add_argument('--dataset', type=int, choices=[1, 2], default=1, help='Dataset option: 1 for dataset/nl/seq2seq/en2de/wmt17_en_de, 2 for dataset/nl/seq2seq/en2fr/wmt14_en_fr')
    parser.add_argument('--finetuning', action='store_true', help='Whether to perform finetuning using the pre-trained model')
    args = parser.parse_args()

    train(args.dataset, args.finetuning)
