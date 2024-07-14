import gzip
import numpy as np
import tqdm
import json
import time
import datetime

from transformers.optimization import get_constant_schedule_with_warmup
from model.optimizer import get_optimizer

import torch
from torch.utils.data import DataLoader

from utils import TextSamplerDataset, MyCollate, ids_to_tokens, BPE_to_eval, epoch_time, count_parameters, mpp_generate_postprocessing

from model.xtransformer import XTransformer

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import sacrebleu

def main(finetuning):

    ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_2 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1, ddp_kwargs_2])

    with open('dataset/nl/wmt17_en_de/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)

    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    EPOCHS = 40
    BATCH_SIZE = 156
    LEARNING_RATE = 1e-4
    GENERATE_EVERY  = 1
    ENC_SEQ_LEN = 120
    DEC_SEQ_LEN = 120
    MAX_LEN = 120
    WARMUP_STEP = 4000

    model = XTransformer(
        dim = 512,
        tie_token_embeds = True,
        return_tgt_loss = True,
        enc_num_tokens=NUM_TOKENS,
        enc_depth = 6,
        enc_heads = 8,
        enc_max_seq_len = ENC_SEQ_LEN,
        dec_num_tokens = NUM_TOKENS,
        dec_depth = 6,
        dec_heads = 8,
        dec_max_seq_len = DEC_SEQ_LEN
    )

    print('number of parameters:', count_parameters(model))

    with gzip.open('dataset/nl/wmt17_en_de/train.en.ids.gz', 'r') as file:
        X_train = file.read()
        X_train = X_train.decode(encoding='utf-8')
        X_train = X_train.split('\n')
        X_train = [np.array([int(x) for x in line.split()]) for line in X_train]

    with gzip.open('dataset/nl/wmt17_en_de/train.de.ids.gz', 'r') as file:
        Y_train = file.read()
        Y_train = Y_train.decode(encoding='utf-8')
        Y_train = Y_train.split('\n')
        Y_train = [np.array([int(x) for x in line.split()]) for line in Y_train]

    with gzip.open('dataset/nl/wmt17_en_de/valid.en.ids.gz', 'r') as file:
        X_dev = file.read()
        X_dev = X_dev.decode(encoding='utf-8')
        X_dev = X_dev.split('\n')
        X_dev = [np.array([int(x) for x in line.split()]) for line in X_dev]

    with gzip.open('dataset/nl/wmt17_en_de/valid.de.ids.gz', 'r') as file:
        Y_dev = file.read()
        Y_dev = Y_dev.decode(encoding='utf-8')
        Y_dev = Y_dev.split('\n')
        Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev]


    train_dataset = TextSamplerDataset(X_train, Y_train, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=4, shuffle=True,
                           pin_memory=True, collate_fn=MyCollate(pad_idx=3))
    dev_dataset = TextSamplerDataset(X_dev, Y_dev, MAX_LEN)
    dev_loader  = DataLoader(dev_dataset, batch_size=1)

    # with gzip.open('dataset/nl/wmt17_en_de/valid.en.ids.gz', 'r') as file:
    #     X_dev = file.read()
    #     X_dev = X_dev.decode(encoding='utf-8')
    #     X_dev = X_dev.split('\n')
    #     X_dev = [np.array([int(x) for x in line.split()]) for line in X_dev]
    #     X_dev = X_dev[0:10]
    #
    # with gzip.open('dataset/nl/wmt17_en_de/valid.de.ids.gz', 'r') as file:
    #     Y_dev = file.read()
    #     Y_dev = Y_dev.decode(encoding='utf-8')
    #     Y_dev = Y_dev.split('\n')
    #     Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev]
    #     Y_dev = Y_dev[0:10]
    #
    # train_dataset = TextSamplerDataset(X_dev, Y_dev, MAX_LEN)
    # train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=4, shuffle=True,
    #                        pin_memory=True, collate_fn=MyCollate(pad_idx=3))
    # dev_dataset = TextSamplerDataset(X_dev, Y_dev, MAX_LEN)
    # dev_loader  = DataLoader(dev_dataset, batch_size=1)



    # optimizer
    optimizer = get_optimizer(model.parameters(), LEARNING_RATE, wd=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEP)

    model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)

    if finetuning:
        print('finetune')
        model.load_state_dict(
            torch.load(
                'output/model_seq2seq.pt',
            ),
        )

    report_loss = 0.
    best_bleu = 0

    # training
    for i in tqdm.tqdm(range(EPOCHS), desc='training'):
        start_time = time.time()
        model.train()

        countdown = 0

        for src, tgt in train_loader:

            mask_src = src != 3

            countdown += 1

            loss = model(src, tgt.type(torch.LongTensor), mask_src=mask_src)

            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            report_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print('[Epoch %d] epoch elapsed %ds' % (i, time.time() - start_time))

        log_str = '[EPOCH %d] loss_train=%.5f' % (i, report_loss/countdown)

        print(log_str)

        report_loss = 0

        torch.save(model.state_dict(),
                   'output/model_seq2seq_each_epoch.pt'
                   )

        torch.save(optimizer.state_dict(), 'output/optim_seq2seq_each_epoch.bin')

        if i != 0 and i % GENERATE_EVERY == 0:

            model.eval()
            target = []
            predicted = []
            for src, tgt in dev_loader:
                start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

                sample = model.module.generate(src, start_tokens, MAX_LEN)

                sample = mpp_generate_postprocessing(sample, eos_token=0)

                # print(f"input:  ", src)
                # print(f"target:", tgt)
                # print(f"predicted output:  ", sample)

                target.append(ids_to_tokens(tgt.tolist()[0], vocabulary))
                predicted.append(ids_to_tokens(sample.tolist()[0], vocabulary))

            target_bleu = [BPE_to_eval(sentence) for sentence in target]

            predicted_bleu = [BPE_to_eval(sentence) for sentence in predicted]

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            bleu = sacrebleu.corpus_bleu(predicted_bleu, [target_bleu])
            bleu = bleu.score
            print('Epoch: {0} | Time: {1}m {2}s, bleu score = {3}'.format(i, epoch_mins, epoch_secs, bleu))

            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(model.state_dict(),
                           'output/model_seq2seq.pt'
                           )

                torch.save(optimizer.state_dict(), 'output/optim_seq2seq.bin')


def test():

    ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_2 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1, ddp_kwargs_2])

    with open('dataset/nl/wmt17_en_de/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)

    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants
    ENC_SEQ_LEN = 120
    DEC_SEQ_LEN = 120
    MAX_LEN = 120 * 2


    with gzip.open('dataset/nl/wmt17_en_de/test.en.ids.gz', 'r') as file:
        X_dev = file.read()
        X_dev = X_dev.decode(encoding='utf-8')
        X_dev = X_dev.split('\n')
        X_dev = [np.array([int(x) for x in line.split()]) for line in X_dev]

    with gzip.open('dataset/nl/wmt17_en_de/test.de.ids.gz', 'r') as file:
        Y_dev = file.read()
        Y_dev = Y_dev.decode(encoding='utf-8')
        Y_dev = Y_dev.split('\n')
        Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev]

    test_dataset = TextSamplerDataset(X_dev, Y_dev, MAX_LEN)
    test_loader  = DataLoader(test_dataset, batch_size=1)

    model = XTransformer(
        dim=512,
        tie_token_embeds=True,
        return_tgt_loss=True,
        enc_num_tokens=NUM_TOKENS,
        enc_depth=6,
        enc_heads=8,
        enc_max_seq_len=ENC_SEQ_LEN,
        dec_num_tokens=NUM_TOKENS,
        dec_depth=6,
        dec_heads=8,
        dec_max_seq_len=DEC_SEQ_LEN
    )

    model, test_loader = accelerator.prepare(model, test_loader)

    model.load_state_dict(
        torch.load(
            'output/model_seq2seq_each_epoch.pt',
        ),
    )

    model.eval()
    target = []
    predicted = []
    for src, tgt in test_loader:
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        sample = model.module.generate(src, start_tokens, MAX_LEN)

        # start_tokens = (torch.ones((1, 1)) * 1).long()
        #
        # sample = model.generate(src, start_tokens, MAX_LEN)

        sample = mpp_generate_postprocessing(sample, eos_token=0)

        print(f"input:  ", ids_to_tokens(src.tolist()[0][1:-1], vocabulary))
        print(f"target:", ids_to_tokens(tgt.tolist()[0][1:-1], vocabulary))
        print(f"predicted output:  ", ids_to_tokens(sample.tolist()[0][:-1], vocabulary))

        target.append(ids_to_tokens(tgt.tolist()[0][1:-1], vocabulary))
        predicted.append(ids_to_tokens(sample.tolist()[0][:-1], vocabulary))

    target_bleu = [BPE_to_eval(sentence) for sentence in target]

    predicted_bleu = [BPE_to_eval(sentence) for sentence in predicted]

    bleu = sacrebleu.corpus_bleu(predicted_bleu, [target_bleu])
    bleu = bleu.score

    print('bleu test equal = ', bleu)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the model", action="store", default=False)
    parser.add_argument("--test", help="test the model", action="store", default=False)

    args = parser.parse_args()

    # Now you can access the input and output file names as follows:
    is_training = args.train
    is_testing = args.test

    if eval(is_training):
        print("training mode")
        main(finetuning=False)
    if eval(is_testing):
        print("testing mode")
        test()
