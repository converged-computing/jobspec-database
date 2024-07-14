import os
import math
import re
import random
import shutil
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn import metrics

from scipy.stats import norm
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
# %matplotlib inline
from matplotlib.colors import LinearSegmentedColormap
from toolz import interleave
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import Accuracy
# from torchsummary import summary
import cv2

# /home/tuk76325/work/PythonProjects/MinaAutoEncoder

print(torch.cuda.device_count())

for i in range(torch.cuda.device_count()-1):
   print(torch.cuda.get_device_properties(i).name)

ALPHA = "-ACDEFGHIKLMNPQRSTVWY"
SEQ_DEPTH = len(ALPHA) #21 alpha len
ROOT_PATH = f"/home"
DATA_FOLDER = "/tuk76325/work/PythonProjects/MinaAutoEncoder"
N_POS = "43pos"
train_sequences_file_name = f"/train_PF00018uniprot_filtered0.2"
test_sequences_file_name = f"/test_PF00018uniprot_filtered0.2"
valid_sequences_file_name = f"/val_PF00018uniprot_filtered0.2"


def load_seqs(file_path, alpha=ALPHA):
    seqs = []
    with open(file_path, 'r') as fin:
        for line in fin:
            seqs.append(list(map(lambda x: alpha.index(x), list(line[:-1]))))
    return np.array(seqs)


# train_seqs = load_seqs(ROOT_PATH + os.path.sep + DATA_FOLDER + os.path.sep + train_sequences_file_name,
#                        alpha=ALPHA).astype("int32")
# val_seqs = load_seqs(ROOT_PATH + os.path.sep + DATA_FOLDER + os.path.sep + valid_sequences_file_name,
#                      alpha=ALPHA).astype("int32")
# test_seqs = load_seqs(ROOT_PATH + os.path.sep + DATA_FOLDER + os.path.sep + test_sequences_file_name,
#                       alpha=ALPHA).astype("int32")
train_seqs = load_seqs(ROOT_PATH + DATA_FOLDER + train_sequences_file_name,
                       alpha=ALPHA).astype("int32")
val_seqs = load_seqs(ROOT_PATH + DATA_FOLDER + valid_sequences_file_name,
                     alpha=ALPHA).astype("int32")
test_seqs = load_seqs(ROOT_PATH + DATA_FOLDER + test_sequences_file_name,
                      alpha=ALPHA).astype("int32")
train_seqs.shape, train_seqs[0]

print(f"Data shape: {train_seqs.shape} {val_seqs.shape}")
SEQ_LEN = train_seqs.shape[-1] #43 = seq len


# Model
class Encoder(nn.Module):
    def __init__(self, hidden_dims=[200,100],
                 latent_dim=6,
                 seq_len=SEQ_LEN,
                 seq_depth=SEQ_DEPTH,
                 activation=None):
        super(Encoder, self).__init__()

        '''
        Use nn.ModuleList if you want to generate a variable number of layers
        and apply them manually in a for loop inside the forward pass
        '''
        # self.linear_layers = nn.ModuleList([for _ in range(num_layers)])
    
        if hidden_dims:
            # print("hidden dimes")
            # print(hidden_dims)
            self.first_layer = nn.Linear(seq_len * seq_depth, hidden_dims[0]) #Input Layer Applies a linear transformation to the incoming data: y=xAT+b (903,100)
            rest_of_layers = []
            for i in range(len(hidden_dims)):
                rest_of_layers.append(
                    nn.Linear(int(hidden_dims[i]), latent_dim if i == len(hidden_dims) - 1 else int(hidden_dims[i + 1])))
                if activation:
                    rest_of_layers.append(activation())

            self.rest_of_layers = nn.Sequential(*rest_of_layers) if rest_of_layers else nn.Linear(int(hidden_dims[-1]),
                                                                                                  latent_dim)
        else:
            self.first_layer = nn.Linear(seq_len * seq_depth, latent_dim)
            self.rest_of_layers = None
    
        self.first_activation = activation() if activation else None
        
        
    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear)):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        x = torch.reshape(inputs, (inputs.shape[0], -1))
        x = self.first_layer(x)
        # print('x shape first')
        # print(x.shape) #100,200
        if self.first_activation:
            x = self.first_activation(x)
        if self.rest_of_layers:
            x = self.rest_of_layers(x)
            # print('x shape rest')
            # print(x.shape) #100, 40
            # print('self rest layers encoder')
            # print(self.rest_of_layers) #(0): Linear(in_features=200, out_features=100, bias=True) ||| (1): Linear(in_features=100, out_features=40, bias=True)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dims=[100,200],
                 latent_dim=100,
                 seq_len=SEQ_LEN,
                 seq_depth=SEQ_DEPTH,
                 activation=None):
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.seq_len = seq_len
        self.seq_depth = seq_depth
        if hidden_dims:
            self.first_layer = nn.Linear(latent_dim, hidden_dims[0])
            # print(latent_dim) #prints 40
            # print(hidden_dims) #prints [100,200]
            self.first_activation = activation() if activation else None
            hidden_layers = []

            for i in range(len(hidden_dims)):
                hidden_layers.append(nn.Linear(int(hidden_dims[i]), seq_len*seq_depth if i == len(hidden_dims) - 1 else int(hidden_dims[i + 1]))) #left off here trying to figure out why i never = 1
                print(str(i) + ' count')
                # hidden_layers.append(nn.Linear(hidden_dims[i + 1],seq_len * seq_depth if i == len(hidden_dims) - 2 else hidden_dims[
                #                                    i + 2]))
                if activation and i < len(hidden_dims) - 2:
                    hidden_layers.append(activation())

            self.middle_layers = nn.Sequential(*hidden_layers) if hidden_layers else nn.Linear(hidden_dims[0],
                                                                                               seq_len * seq_depth)

        else:
            self.last_layer = nn.Linear(latent_dim, seq_len * seq_depth)
        self.last_activation = nn.Softmax(-1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear)):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        if self.hidden_dims:
            # print('inputs')
            # print(inputs.shape) #torch.Size([100, 40])
            x = self.first_layer(inputs)
            # print('x self first layer')
            # print(x.shape) #torch.Size([100, 100])
            if self.first_activation:
                x = self.first_activation(x)
            if self.middle_layers:
                # print('self middle layers')
                # print(self.middle_layers) # (0): Linear(in_features=100, out_features=200, bias=True)
                # print('x self middle layer') 
                x = self.middle_layers(x) 
                # print(x.shape)
        else:
            x = self.last_layer(inputs)
        x = torch.reshape(x, (-1, self.seq_len, self.seq_depth)) #what is the point of this is it for the activation fxn? its giving an error 
        x = self.last_activation(x)
        return x


# encoder = Encoder(hidden_dims=[],
#                   latent_dim=6, activation=nn.ReLU).to('cuda')
# # summary(encoder, input_size = (SEQ_DEPTH, SEQ_LEN), batch_size=-1)



# decoder = Decoder(hidden_dims=[],
#                   latent_dim=6, activation=nn.ReLU).to('cuda')


# summary(decoder, input_size = (6,), batch_size=-1)

# Model Wrapper

class LVAE(L.LightningModule):
    def __init__(self, latent_dim_size=100,
                 encoder_hidden_dims=[],
                 decoder_hidden_dims=[],
                 activation=None,
                 input_len=SEQ_LEN,
                 sequence_depth=SEQ_DEPTH,
                 learning_rate=2e-4,
                 gamma=0.99):
        super().__init__()
        self.sequence_len = input_len
        self.sequence_depth = sequence_depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.encoder = Encoder(hidden_dims=encoder_hidden_dims,
                               latent_dim=latent_dim_size,
                               activation=activation,
                               seq_len=input_len,
                               seq_depth=sequence_depth)
        self.decoder = Decoder(hidden_dims=decoder_hidden_dims,
                               latent_dim=latent_dim_size,
                               seq_len=input_len,
                               seq_depth=sequence_depth,
                               activation=activation)
        self.metric = MulticlassAccuracy(num_classes=sequence_depth,
                                         average='micro')
        self.metric_fn = Accuracy(task="multiclass", num_classes=sequence_depth)
        self.rec_fn = nn.CrossEntropyLoss()

    # def reparameterize(self, mu, logvar):
    #   """
    #   Reparameterization trick to sample from N(mu, var) from
    #   N(0,1).
    #   :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #   :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #   :return: (Tensor) [B x D]
    #   """
    #   std = torch.exp(0.5 * logvar)
    #   eps = torch.randn_like(std)
    #   return eps * std + mu

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return [recon]

    def loss_function(self, target, *args):
        recons = args[0]

        recon_loss = self.rec_fn(recons.transpose(1, 2), target.transpose(1, 2))

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recon_loss  # + kld_loss
        acc = self.metric_fn(torch.argmax(recons, dim=-1),
                             torch.argmax(target, dim=-1))

        return loss, acc  # , recon_loss, kld_loss,

    def training_step(self, batch, batch_idx):
        input, target_seq = batch
        results = self.forward(input)
        loss, acc = self.loss_function(target_seq, *results)
        # self.log('reconstruction', loss, sync_dist=True)
        # self.log('acc', acc, sync_dist=True)
        self.log_dict({
            'loss': loss,
            'acc': acc
        }, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input, target_seq = batch
        results = self.forward(input)
        loss, acc = self.loss_function(target_seq, *results)
        # self.log('val_reconstruction', recon_loss, sync_dist=True)
        # self.log('val_acc', acc, sync_dist=True)
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, prog_bar=True)

        return loss

    '''
    enc_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
    dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
    enc_scheduler = torch.optim.lr_scheduler.ExponentialLR(enc_optimizer,
                                                            gamma=self.gamma)
    dec_scheduler = torch.optim.lr_scheduler.ExponentialLR(enc_optimizer,
                                                            gamma=self.gamma)
    return [enc_optimizer, dec_optimizer], [enc_scheduler, dec_optimizer]
    '''

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=self.gamma)
        return [optimizer], [scheduler]

    # Hyper Parameters

def cast_list(test_list, data_type):
    return list(map(data_type, test_list))

Hdim_List = cast_list(sys.argv[1].split(','), int)

# Hdim_List = np.array(sys.argv[1].split(','), dtype=int)
# Hdim_List = list(Hdim_List)

print('hdimlist')
print(Hdim_List)
LATENT_DIM = 40
ENCODER_H_DIMS = Hdim_List
DECODER_H_DIMS = Hdim_List.reverse()
ACTIVATION = None
if sys.argv[2] == 'relu':
    ACTIVATION = nn.ReLU
elif sys.argv[2] == 'leakyrelu':
    ACTIVATION == nn.LeakyReLU
elif sys.argv[2] == 'tanh':
    ACTIVATION == nn.Tanh
elif sys.argv[2] == 'sigmoid':
    ACTIVATION == nn.Sigmoid
else:
    ACTIVATION = None
BATCH_SIZE = 100
EPOCHS = 750
LEARNING_RATE = 2e-4
VERSION = f"[L_torch]aVAE_LD_{LATENT_DIM}_BS_{BATCH_SIZE}_v0.0"
NUM_LAYERS = len(ENCODER_H_DIMS)

# model = LVAE(self, latent_dim_size=LATENT_DIM, encoder_hidden_dims=ENCODER_H_DIMS, decoder_hidden_dims=DECODER_H_DIMS, activation=ACTIVATION)

save_dir = f"work/torch_ckpts/{VERSION}/"

if not os.path.exists(save_dir):
  os.makedirs(save_dir)

#Data Loader 

class MyDataModule(L.LightningDataModule):
  def __init__(self, batch_size=BATCH_SIZE, seq_depth=SEQ_DEPTH):
      super().__init__()
      self.seq_depth = seq_depth
      self.batch_size = batch_size

  class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, seq_depth):
      self.seq_depth = seq_depth
      self.model_in = seqs

    def __len__(self):
      return self.model_in.shape[0]

    def __getitem__(self, idx):
      ds = torch.unsqueeze(torch.from_numpy(self.model_in[idx]), 0).long()
      ds_onehot = F.one_hot(ds, num_classes=self.seq_depth).type(torch.FloatTensor)
      ds_onehot = torch.squeeze(ds_onehot, dim=0)
      return ds_onehot, ds_onehot

  def set_data(self, data):
    self.data = data


  def setup(self, stage):
      # Assign train/val datasets for use in dataloaders
      if stage == 'fit':
        self.train_set = self.SeqDataset(self.data['train'], self.seq_depth)
        self.val_set = self.SeqDataset(self.data['valid'], self.seq_depth)
      if stage == 'test':
          self.test_set = self.SeqDataset(self.data['test'], self.seq_depth)

  def train_dataloader(self):
    return DataLoader(self.train_set, self.batch_size, shuffle=True, drop_last=True, num_workers=2)

  def val_dataloader(self):
    return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=2)

  def test_dataloader(self):
    return DataLoader(self.test_set, self.batch_size, shuffle=False)


torch.cuda.empty_cache()
dm = MyDataModule(BATCH_SIZE, SEQ_DEPTH)
dm.set_data(data={'train': train_seqs, 'valid':val_seqs})
dm.setup(stage='fit')

#Training

device_name = 'cuda'

vae = LVAE(latent_dim_size=LATENT_DIM,
               encoder_hidden_dims=ENCODER_H_DIMS,
               decoder_hidden_dims=DECODER_H_DIMS,
               activation=ACTIVATION,
               learning_rate=LEARNING_RATE)

# opt_vae = torch.compile(vae, mode="reduce-overhead")

trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator=device_name,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=25)],
    default_root_dir=save_dir,
    # logger=logger,
    # devices=4,
    # log_every_n_steps=5,
)

# Train
trainer.fit(vae, dm)
vae.eval()
y_one_hot_encoded = F.one_hot(torch.from_numpy(test_seqs).long(), num_classes=SEQ_DEPTH).type(torch.FloatTensor)

with torch.no_grad():
    y_hat = vae(y_one_hot_encoded)[0]
y_hat = np.argmax(y_hat.numpy(), axis=-1)
print(y_hat.shape)

test_acc = metrics.accuracy_score(test_seqs.flatten(), y_hat.flatten())

# columns = ["EPOCHS", "LD", "Acc"]
# vals = np.array([EPOCHS, LATENT_DIM, test_acc]).reshape((1, -1))

with open('Output.txt', 'a') as fp:
    # fp.write("EPOCHS, LD, Acc \n")
    fp.write(f"{EPOCHS}, {NUM_LAYERS}, {LATENT_DIM}, {ENCODER_H_DIMS}, {sys.argv[2]}, {test_acc} \n")
fp.close()


#Deletes repeat line of statistics

with open("Output.txt", 'r') as fp:
    lines = fp.readlines()  #all lines in a list
fp.close()

with open('Output.txt', 'w') as fp:
    for number, line in enumerate(lines):
        if number not in [len(lines)-1]:
            fp.write(line)
fp.close()

# vals.tofile('SingleLayerNoActivation.csv', sep = ',')

# df = pd.DataFrame(data=vals, columns=columns)

# df.to_csv(ROOT_PATH + os.sep + "tuk76325" + os.sep + "output" + os.sep + "SingleLayerNoActivation.csv", mode='a')