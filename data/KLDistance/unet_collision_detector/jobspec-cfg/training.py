import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNET
from dataset import DataComposer, SampleDataSet, RealDataComposer

class Training:
    def __init__(self, train_data_loader, val_data_loader, device='cpu', epochs=100, load_model=False, thresh=0.5):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.epochs = epochs
        self.thresh = thresh
    def save_checkpoint(self, mdl_stat, file_name='./model/training_model/default_checkpoint.stat'):
        print('\n===> saving checkpoint <===')
        torch.save(mdl_stat, file_name)
    def load_checkpoint(self, mdl, file_name='./model/training_model/default_checkpoint.stat'):
        print('\n===> loading checkpoint <===')
        mdl.load_state_dict(torch.load(file_name)['state_dict'])
    def val_fn(self, model):
        num_correct = 0
        num_pixels = 0
        model.eval()
        with torch.no_grad():
            for x, y in self.val_data_loader:
                x = x.float().to(device=self.device)
                y = y.float().to(device=self.device)
                preds = torch.sigmoid(model(x))
                preds = (preds > self.thresh).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
        print(f'validation {num_correct}/{num_pixels}, acc: {num_correct/num_pixels*100:.2f}%.')
        model.train()
    def train_fn(self, model, optimizer, loss_fn, scaler):
        loop = tqdm(self.train_data_loader)
        for batch_idx, (signal, label) in enumerate(loop):
            signal = signal.float().to(device=self.device)
            label = label.float().to(device=self.device)
            # forward
            if self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    predictions = model(signal)
                    loss = loss_fn(predictions, label)
            else:
                predictions = model(signal)
                loss = loss_fn(predictions, label)
            # backward
            if self.device == 'cuda':
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # update loop
            loop.set_postfix(loss=loss.item())
    def iterated_training(self, model, optimizer, loss_fn, scaler, load_model=False, load_model_name='default_checkpoint.stat', tar_model_name='default_checkpoint.stat'):
        if load_model:
            self.load_checkpoint(mdl=model, file_name=load_model_name)
        for epoch in range(self.epochs):
            print(f'--- Epoch: {epoch} ---')
            self.train_fn(model, optimizer, loss_fn, scaler)
            # save model
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            self.save_checkpoint(checkpoint, file_name=tar_model_name)
            # check accuracy
            self.val_fn(model)

def fake_training_routine(epochs, load_model=False, load_model_name='default_checkpoint.stat', tar_model_name='default_checkpoint.stat'):
    # data generation
    data_composer = DataComposer(file_name='./data/training_dataset/train1', sample_num=400)
    data_composer.compose()
    data_composer.save()
    data_composer = DataComposer(file_name='./data/training_dataset/val1', sample_num=100)
    data_composer.compose()
    data_composer.save()
    # dataset and data loader
    train_dataset = SampleDataSet(data_path='./data/training_dataset/train1')
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True)
    val_dataset = SampleDataSet(data_path='./data/training_dataset/val1')
    val_data_loader = DataLoader(val_dataset, batch_size=32, num_workers=1, shuffle=False)
    # check cuda status
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    is_load_model = True
    # model
    model = UNET()
    # loss function
    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=4e-6)
    # scaler
    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        model.to(device='cuda')
    else:
        scaler = None
    # training
    training = Training(train_data_loader=train_data_loader, val_data_loader=val_data_loader, device=device, epochs=epochs)
    training.iterated_training(model, optimizer, loss_fn, scaler, load_model=load_model, load_model_name=load_model_name, tar_model_name=tar_model_name)

def real_blank_training_routine(epochs, load_model=False, load_model_name='default_checkpoint.stat', tar_model_name='default_checkpoint.stat'):
    # data generation
    data_composer = RealDataComposer(src_name_list=['./experimental_data/12192022_blanks/SA3_215_FcBackround_B_compilednoname.csv'], 
    tar_name='./data/training_dataset/blank_real_train1')
    data_composer.compose()
    data_composer.save()
    data_composer = RealDataComposer(src_name_list=['./experimental_data/12192022_blanks/SA3_215_FcBackround_A_compilednoname.csv'], 
    tar_name='./data/training_dataset/blank_real_val1')
    data_composer.compose()
    data_composer.save()
    # dataset and data loader
    train_dataset = SampleDataSet(data_path='./data/training_dataset/blank_real_train1')
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True)
    val_dataset = SampleDataSet(data_path='./data/training_dataset/blank_real_val1')
    val_data_loader = DataLoader(val_dataset, batch_size=32, num_workers=1, shuffle=False)
    # check cuda status
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    is_load_model = True
    # model
    model = UNET()
    # loss function
    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # scaler
    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        model.to(device='cuda')
    else:
        scaler = None
    # training
    training = Training(train_data_loader=train_data_loader, val_data_loader=val_data_loader, device=device, epochs=epochs)
    training.iterated_training(model, optimizer, loss_fn, scaler, load_model=load_model, load_model_name=load_model_name, tar_model_name=tar_model_name)

if __name__ == '__main__':
    fake_counts = 20
    fake_epoch = 80
    real_counts = 1
    real_epoch = 10
    load_model = True
    src_name = './model/training_model/fine_tunning_01062023_3.stat'
    tar_name = './model/training_model/fine_tunning_01062023_4.stat'
    for iter in range(fake_counts):
        print(f'---- fake training ({iter+1}/{fake_counts}) ----')
        if iter == 0:
            fake_training_routine(fake_epoch, load_model, src_name, tar_name)
        else:
            fake_training_routine(fake_epoch, load_model, tar_name, tar_name)
    for iter in range(real_counts):
        print(f'---- real training ({iter+1}/{real_counts}) ----')
        real_blank_training_routine(real_epoch, load_model, tar_name, tar_name)