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
from src.model.adafactor import Adafactor

import os
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import copy
from src.model.model import MultiHeadedAttention, PositionwiseFeedForward, \
    PositionalEncoding, Encoder, EncoderLayer, Generator, Embeddings
import torch.nn as nn

writer = SummaryWriter()

class Critic(nn.Module):

    def __init__(self, encoder, src_embed, generator):
        super(Critic, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
        self.steps = 0

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        x = self.src_embed(x)
        for layer in self.encoder.layers:
            x = layer(x, mask)
        return self.encoder.norm(x)


def make_critic(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyper-parameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    generator = Generator(d_model, tgt_vocab)
    embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    critic = Critic(encoder, embed, generator)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in critic.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return critic


t = time.time()
last_saved = t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dae_input(tgt, token_selector):
    select_prob_embeds = token_selector.forward(tgt.to(device),
                                         (tgt != 3).unsqueeze(-2).to(device))
    select_prob = token_selector.generator(select_prob_embeds)
    select_indices = torch.max(select_prob, dim=2).indices.type(torch.ByteTensor)
    dae_list = []
    for ind, row in zip(select_indices, tgt):
        dae_list.append(torch.masked_select(row, ind)[:15])
    dae_input = torch.nn.utils.rnn.pad_sequence(dae_list, batch_first=False, padding_value=3)
    return dae_input


def run_epoch(data_iter, model, token_selector, loss_compute, tokenizer, token_optim,
              save_path=None, validate=False, criterion=None, model_opt=None, acc_steps=8):
    """Standard Training and Logging Function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global t, last_saved
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        try:
            tgt, tgt_mask = batch.trg.to(device), batch.trg_mask.to(device)
            # classify tokens, get the first 15 tokens selected.
            dae_input = get_dae_input(batch.trg, token_selector).transpose(0, 1).to(device)
            # create src and src mask from selected tokens
            dae_input_mask = (dae_input != 3).unsqueeze(-2)

            # get output of poetry generator
            output_embeds = model.forward(dae_input, tgt, dae_input_mask, tgt_mask)
            output = model.generator(output_embeds)

        except RuntimeError:
            print("OOM - skipping batch", i, "SRC shape:", batch.src.shape(), "TGT shape:", batch.tgt.shape())
            continue
        reconstruction_loss = criterion(output.contiguous().view(-1, output.size(-1)),
                                              batch.trg_y.to(device).contiguous().view(
                                                  -1)) / batch.ntokens

        writer.add_scalar(exp_name + "/Loss", float(reconstruction_loss) , global_step=model.steps)
        writer.add_scalar(exp_name + "/Learning Rate", loss_compute.opt._rate, global_step=model.steps)
        if model_opt is not None:
            reconstruction_loss.backward()
            if i % acc_steps == 0:
                model_opt.step()
                model_opt.optimizer.zero_grad()
                token_optim.step()
                token_optim.zero_grad()
        total_loss += float(reconstruction_loss)
        del output
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
                        'optimizer_state_dict': model_opt.optimizer.state_dict(),
                        'selector_state_dict': token_selector.state_dict(),
                        'selector_optim_state_dict': token_optim.state_dict()
                    },
                        save_file)
                    last_saved = time.time()
                    if torch.cuda.is_available():
                        qsub(save_file, model.steps)
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, reconstruction_loss / ntokens, tokens / elapsed))
            # sanity_check()
            start = time.time()
            tokens = 0
        del reconstruction_loss
        torch.cuda.empty_cache()
    return total_loss / total_tokens


def run_training(dataset, tokenizer, epochs=1000000, vocab_size=32000, config_name=None, acc_steps=8):
    bsz = 4000 if dataset == "cz" else 4500
    train_iter, valid_iter, test_iter, train_idx, dev_idx, test_idx = get_training_iterators(dataset, batch_size=bsz)
    if config_name is None:
        config_name = "dae"
    save_path = "checkpoints/" + dataset + "-" + config_name
    pad_idx = 3
    model = make_model(vocab_size, vocab_size, N=6).to(device)
    token_selector = make_critic(vocab_size, 2, N=2).to(device)
    criterion = LabelSmoothing(size=vocab_size, padding_idx=pad_idx, smoothing=0.1)
    token_optim = Adafactor(token_selector.parameters())
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        last = sorted(os.listdir(save_path), reverse=True, key=lambda x: int(x.partition(".")[0]))[0]
        last_file = os.path.join(save_path, last)
        checkpoint = torch.load(last_file)
        token_selector.load_state_dict(checkpoint['selector_state_dict'])
        token_optim.load_state_dict(checkpoint['selector_optim_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model_opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.steps = int(last.split(".")[0])
    else:
        model.steps = 0

    print("Training with 1 GPU.")
    model = model.to(device)
    loss_train = SimpleLossCompute(model.generator, criterion, model_opt)
    for epoch in range(epochs):
        model.train()
        run_epoch(data_iter=(rebatch(pad_idx, b) for b in train_iter), model=model, loss_compute=loss_train,
                  token_selector=token_selector, tokenizer=tokenizer, token_optim=token_optim,
                  model_opt=model_opt, criterion=criterion, save_path=save_path, acc_steps=acc_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tur")
    parser.add_argument("--acc_steps", default=8)
    args = parser.parse_args()
    dataset_lang = {"tur": "tr", "eng": "en", "cz": "cz"}
    tokenizer = get_tokenizer(dataset_lang[args.dataset])
    run_training(args.dataset, tokenizer, args.acc_steps)

