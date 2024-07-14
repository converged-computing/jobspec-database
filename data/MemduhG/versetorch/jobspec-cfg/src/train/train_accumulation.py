#!/storage/praha1/home/memduh/versetorch/venv python
import argparse
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils.batch import rebatch
from src.data_utils.data import get_training_iterators
from src.model.loss_optim import MultiGPULossCompute, SimpleLossCompute
from src.model.model import make_model, NoamOpt, LabelSmoothing, translate_sentence
from src.utils.utils import get_tokenizer
from src.utils.qsub import qsub

import os
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

t = time.time()
last_saved = t
writer = SummaryWriter()


def run_epoch(data_iter, model, loss_compute, tokenizer, save_path=None, validate=False,
              criterion=None, model_opt=None, acc_steps=8, exp_name=None):
    """Standard Training and Logging Function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global t, last_saved
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        try:
            out = model.forward(batch.src.to(device), batch.trg.to(device),
                             batch.src_mask.to(device), batch.trg_mask.to(device))
        except RuntimeError:
            print("OOM - skipping batch", i, "SRC shape:", batch.src.shape(), "TGT shape:", batch.tgt.shape())
            continue

        x = model.generator(out)
        loss = criterion(x.contiguous().view(-1, x.size(-1)), batch.trg_y.to(device).contiguous().view(-1)) / batch.ntokens
        
        writer.add_scalar(exp_name + "/Loss", float(loss) , global_step=model.steps)
        writer.add_scalar(exp_name + "/Learning Rate", loss_compute.opt._rate, global_step=model.steps)
        if model_opt is not None:
            loss.backward()
            if i % acc_steps == 0:
                model_opt.step()
                model_opt.optimizer.zero_grad()
        total_loss += float(loss)
        del out
        ntokens = batch.ntokens
        total_tokens += ntokens
        tokens += ntokens
        del batch
        if loss_compute is not None:
            model.steps += 1
            if save_path is not None:
                if (time.time() - last_saved > 1800) or (not os.path.exists(save_path)) \
                        or len(os.listdir(save_path)) == 0:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_file = save_path + "/" + str(model.steps) + ".pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_opt.optimizer.state_dict()},
                        save_file)
                    last_saved = time.time()
                    if torch.cuda.is_available():
                        qsub(save_file, model.steps)
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / ntokens, tokens / elapsed))
            # sanity_check()
            start = time.time()
            tokens = 0
        del loss
        torch.cuda.empty_cache()
    return total_loss / total_tokens


def run_training(dataset, tokenizer, epochs=1000000, vocab_size=32000, config_name=None, acc_steps=8):
    bsz = 4000 if dataset == "cz" else 4500
    train_iter, valid_iter, test_iter, train_idx, dev_idx, test_idx = get_training_iterators(dataset, batch_size=bsz)
    if config_name is None:
        config_name = "acc"
    save_path = "checkpoints/" + dataset + "-" + config_name
    exp_name = dataset + "-" + config_name
    pad_idx = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(vocab_size, vocab_size, N=6).to(device)
    criterion = LabelSmoothing(size=vocab_size, padding_idx=pad_idx, smoothing=0.1)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        last = sorted(os.listdir(save_path), reverse=True, key=lambda x: int(x.partition(".")[0]))[0]
        last_file = os.path.join(save_path, last)
        checkpoint = torch.load(last_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.steps = int(last.split(".")[0])
    else:
        model.steps = 0

    print("Training with 1 GPU.")
    model = model.to(device)
    loss_train = SimpleLossCompute(model.generator, criterion, model_opt)
    loss_val = SimpleLossCompute(model.generator, criterion, None)
    for epoch in range(epochs):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model, loss_train, tokenizer, model_opt=model_opt,
                  criterion=criterion, save_path=save_path, exp_name=exp_name)
        #model.eval()
        #loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model, loss_val, tokenizer,
        #                 criterion=criterion, validate=True)
        #print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tur")
    parser.add_argument("--acc_steps", default=8)
    args = parser.parse_args()
    dataset_lang = {"tur": "tr", "eng": "en", "cz": "cz", "tur-lower": "tr", "eng-lower": "en", "cz-lower": "cz"}
    tokenizer = get_tokenizer(dataset_lang[args.dataset])
    run_training(args.dataset, tokenizer, args.acc_steps)

