import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

from argparse import ArgumentParser
import random, sys, math, gzip, os
from tqdm import tqdm

from dataloader.nsynthdataset import NSynthDataSet
from util import util
from transformer.transformers import GTransformer


config = util.get_config()
data_dir = config['data_dir']
sample_rate = config['sample_rate']
batch_size = config['batch_size']
lr = config['lr']
lr_warmup = config['lr_warmup']
dropout = config['dropout']
epochs = config['epochs']

sample_length = config['sample_length']
embedding_size = config['embedding_size'] 
num_heads = config['num_heads']
depth = config['depth']
num_tokens = config['num_tokens']

lower_pitch_limit = config['lower_pitch_limit']
upper_pitch_limit = config['upper_pitch_limit']
checkpoint_location = config['checkpoint_location']

print(config)

train_ds = NSynthDataSet(data_dir=data_dir, sr=sample_rate, sample_length=sample_length, split='train')
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = NSynthDataSet(data_dir=data_dir, sr=sample_rate, sample_length=sample_length, split='test')
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

model = GTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=sample_length, num_tokens=num_tokens, dropout=dropout)
model = model.cuda()

opt = torch.optim.Adam(lr=lr, params=model.parameters())
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0), verbose=False)
loss = torch.nn.NLLLoss(reduction='mean')

print(model)

def train():
    training_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
        opt.zero_grad()
        b, cols, seq_len = data.shape
        data = data.cuda().float()
        target = target.cuda()
        
        output = model(data)
        running_loss = loss(output.transpose(2, 1), target)
        training_loss += running_loss.item()

        running_loss.backward() # backward pass
        
        gradient_clipping = 1.0
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        opt.step() 
    
    sch.step()
    training_loss /= len(train_loader)
    print(f'Epoch training loss = {training_loss}, Epoch last LR = {sch.get_last_lr()}', flush=True)
    return training_loss

def test():
    model.eval()
    testing_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc='Testing')):
            data = data.cuda().float()
            target = target.cuda()
            
            output = model(data)

            running_loss = loss(output.transpose(2, 1), target)
            testing_loss += running_loss.item()

    testing_loss /= len(test_loader)
    return testing_loss

history_train = {'loss': []}
history_test = {'loss': []}

for epoch in range(0, epochs, 1):
    train_loss = train()
    history_train['loss'].append(train_loss)
    
    if epoch%1000 == 0 or epoch == epochs-1:
        test_loss = test()
        history_test['loss'].append(test_loss)
        
        fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
        axes[0].plot(history_train['loss'])
        axes[0].set_title('Train Loss')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('loss')
        
        axes[1].plot(history_test['loss'])
        axes[1].set_title('Test Loss')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('loss')
        
        plt.savefig(f'{checkpoint_location}/plots/{epoch}.png')
        
        plt.close(fig)

        util.save_model(epoch, model, opt, train_loss, f'{checkpoint_location}/models/{epoch}.pt')













