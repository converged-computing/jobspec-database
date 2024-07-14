import torch
import torch.nn as nn
from model import conv_audio
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from IPython import embed
import argparse
from FSD_dataloader import FSD18
import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from utils import amc_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from IPython import embed
import random
##### set the seed to reproduce the results#####
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='Train cnn networks')
parser.add_argument('-p', '--params_yaml',
                    dest='params_yaml',
                    action='store',
                    required=False,
                    type=str)
parser.add_argument('-alpha',

                    action='store',
                    required=True,
                    type=float
                    )
args = parser.parse_args()
args.alpha/=10
print(args.alpha)

print('\nYaml file with parameters defining the experiment: %s\n' % str(args.params_yaml))

params = yaml.full_load(open(args.params_yaml))


####### load the dataloader#########



tr_Dataset =FSD18('tr_'+ params['ctrl'].get('train_data'), params['ctrl'].get('dataset_path')+'features/')

training_generator = DataLoader(tr_Dataset,batch_size =params['learn'].get('batch_size'),
                                        shuffle = True,
                                        num_workers = 4,
                                        drop_last = True)

cv_Dataset =FSD18('val_' + params['ctrl'].get('train_data'), params['ctrl'].get('dataset_path')+'features/')

validation_loader = DataLoader(cv_Dataset, batch_size = params['learn'].get('batch_size'),shuffle = True,num_workers = 4, drop_last = True)

model = conv_audio(params['learn'].get('n_classes'))
print(model)


output_dir = './model_'+params['ctrl'].get('train_data')+'/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Directory " , output_dir ,  " Created ")
else:
    print("Directory " , output_dir ,  " already exists")


########### use GPU ##########
use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
########### use GPU ##########


#### define the loss function and the optimizer#########
loss_fn = torch.nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
#scheduler.step()

#### define the loss function and the optimizer#########

print('-----------start training')
######define train function######
def train(epoch,writer_tr):
    model.train()
    train_loss = 0.
    #embed()
    start_time = time.time()
    count  = training_generator.__len__()*(epoch-1)
    loader = tqdm(training_generator)
    for batch_idx, data in enumerate(loader):
        #embed()
        count = count + 1
        batch_embed = data[0].contiguous()
        batch_label = data[1].contiguous()
        #video_name = data[2]
        #embed()
        batch_embed = batch_embed.to(device)
        batch_label = batch_label.to(device)

        # training
        optimizer.zero_grad()
        #import pdb; pdb.set_trace()
       # embed()
        #breakpoint()
        esti_label, fea = model(batch_embed)
        l1 = loss_fn(esti_label,batch_label)
        l2 = amc_loss(fea,batch_label)

        loss = args.alpha* l1 + (1-args.alpha)*l2
        loss.backward()

        train_loss += loss.data.item()
        optimizer.step()


        if (batch_idx+1) % 100 == 0:
            elapsed = time.time() - start_time

            writer_tr.add_scalar('Loss/train', loss.data.item(),count)
            writer_tr.add_scalar('Loss/train_avg', train_loss/(batch_idx+1),count)
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx+1, len(training_generator),
                elapsed * 1000 / (batch_idx+1), loss ))


    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), train_loss))

    return train_loss
######define train function######

###### define validate function######
def validate(epoch,writer_val):
    model.eval()
    validation_loss = 0.
    start_time = time.time()
    # data loading
    for batch_idx, data in enumerate(validation_loader):

        batch_embed =data[0].contiguous()
        batch_label =data[1].contiguous()

        batch_embed = batch_embed.to(device)
        batch_label = batch_label.to(device)

        with torch.no_grad():
             esti_label,fea = model(batch_embed)
             l1 = loss_fn(esti_label,batch_label)
             l2 = amc_loss(fea,batch_label)

             loss = args.alpha*l1 + (1-args.alpha)*l2
             validation_loss += loss.data.item()

    #print('the ',batch_idx,'iteration val_loss is ', validation_loss)
    validation_loss /= (batch_idx+1)
   # embed()
    writer_val.add_scalar('Loss/val', loss.data.item(),batch_idx*epoch)
    writer_val.add_scalar('Loss/val_avg', validation_loss,batch_idx*epoch)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)

    return validation_loss
######define validate function######


training_loss = []
validation_loss = []
decay_cnt = 0
writer_tr = SummaryWriter(os.path.join(output_dir,'train'))
writer_val = SummaryWriter(os.path.join(output_dir,'val'))
for epoch in range(1, params['learn'].get('n_epochs')):
    model.cuda()
    print('this is epoch', epoch)
    training_loss.append(train(epoch, writer_tr)) # Call training
    validation_loss.append(validate(epoch,writer_val)) # call validation



    print('-' * 99)

    if training_loss[-1] == np.min(training_loss):
        print(' Best training model found.')
        print('-' * 99)
        # with open(output_dir+'model_tr.pt', 'wb') as f:
        #     torch.save(model.cpu().state_dict(), f)

        #     print(' Best training model found and saved.')
        #     print('-' * 99)

    if validation_loss[-1] == np.min(validation_loss):
        # save current best model
        with open(output_dir+'model_'+str(args.alpha)+'.pt', 'wb') as f:
            torch.save(model.cpu().state_dict(), f)

            print(' Best validation model found and saved.')
            print('-' * 99)
    scheduler.step(validation_loss[-1])


####### plot the loss and val loss curve####
minmum_val_index=np.argmin(validation_loss)
minmum_val=np.min(validation_loss)
plt.plot(training_loss,'r')
#plt.hold(True)
plt.plot(validation_loss,'b')
plt.axvline(x=minmum_val_index,color='k',linestyle='--')
plt.plot(minmum_val_index,minmum_val,'r*')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig(output_dir+'loss_'+str(args.alpha)+'.png')
####### plot the loss and val loss curve####
