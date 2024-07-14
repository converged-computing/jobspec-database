from data.dataloader import NumpyDataset, NumpyMetaDataset
from utils.utils import Averager, Text2List
import argparse
import time
import os
from argparse import ArgumentParser
import torch
from torch.optim import Adam 
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel  
from torch.autograd import Variable
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm
from nn.lstmae import LSTMAutoencoder, LSTMAutoencoder2, LSTMAutoencoder3 
from nn.tfm import TransformerEncoder
# from nn.crossformer import CrossFormer
from nn.crossodeformer import CrossFormer
from torch.optim.lr_scheduler import StepLR


 

def training_fun(args):  
    key, result_list = time.strftime("contrastive-%Y%m%d-%H%M%S"), []  
    args.save_path = args.save_path   
    device = torch.device("cuda:0")  
    # model = TransformerEncoder(
    #     input_size = 20484, 
    #     d_model = 2048, 
    #     nhead = 8, 
    #     num_layers = 6, 
    #     device = device
    # )
    model = CrossFormer(
        input_dim = 20484,
        dim = 1024, 
        depth = 6, 
        heads = 8, 
        mlp_dim = 1024, 
        dim_head = 64,
        device = device
    )
    model = model.to(device)  
    optimizer = Adam(model.parameters(), lr = 2e-5)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 
    data_embeddings = {}   
    tlf, tlm, tlpn, ta  = Averager(), Averager(), Averager(), Averager()   
    train_dataloader = NumpyMetaDataset(meg_dir = args.meg_dir + 'train/', 
                    fmri_dir =  args.fmri_dir + 'train/', 
                    n_way = 50, 
                    batch_size = args.batch_size, 
                    shuffle=True)  
    
    # test_dataloader = NumpyMetaDataset(meg_dir = args.meg_dir + 'train/', 
    #                 fmri_dir =  args.fmri_dir + 'test/', 
    #                 n_way = 50, 
    #                 batch_size = args.batch_size, 
    #                 shuffle=True)  
    
    model.train()    
    for it, ((xm, xf, y_meta, y_batch), [t, sub_f, sub_m]) in enumerate(train_dataloader):  

        optimizer.zero_grad()  

        xm = torch.stack([torch.from_numpy(arr) for arr in xm]).squeeze(1).to(device)
        xf = torch.stack([torch.from_numpy(arr) for arr in xf]).squeeze(1).to(device)
        xm, xf = Variable(xm.float()), Variable(xf.float()) 

        [loss_pn, loss_m, loss_f], acc_pn, _  = model.forward_contrastive(xm, xf)    
        loss =  loss_m + loss_f + loss_pn
        
        loss.backward()
        optimizer.step()  

        tlf.add(loss_f.item()) 
        tlm.add(loss_m.item()) 
        tlpn.add(loss_pn.item()) 
        ta.add(acc_pn)  

        # if epoch%10==0:
        #     data_embeddings[subj] = [time, z_hm.detach().cpu().numpy(), z_hf.detach().cpu().numpy()]

        if it>0 and it%25 ==0:
            prt_text =  '(id:'+ key + ')    it.%d       l(a/f/m): /%4.2f/%4.2f/%4.2f/ || a: %4.2f' 
            print(prt_text % (it, tlpn.item()*100, tlf.item()*100, tlm.item()*100, ta.item()*100)) 
            
            result_list.append(prt_text % (it, tlpn.item()*100, tlf.item()*100, tlm.item()*100, ta.item()*100)) 
            Text2List('./mar31-0017-mccleary-odeUpDown', result_list)
            tlf, tlm, tlpn, ta  = Averager(), Averager(), Averager(), Averager()   
            # if epoch%10==0:
            #     np.save(f'./results/data_embeddings_{epoch}.npy', data_embeddings) 

        if it>0 and it%500 == 0:
            if not os.path.exists(os.path.join(args.save_path, key)): 
                os.mkdir(os.path.join(args.save_path, key)) 
            pwd = args.save_path+'/'+ key+'/ith_'+str(it)+'_model'+key+'.pth' 
            torch.save({'iteration': it,
                        'args': args,
                        'model': model.state_dict(), 
                        # 'optimizer': optimizer, 
                        'result_list': result_list},
                        pwd) 
            # scheduler.step()

 
 

if __name__ == "__main__": 

    parser = ArgumentParser(add_help=True) 

    parser.add_argument("--model", default="contrastive_pn", type=str)
    parser.add_argument("--save_path", default='./outputs/train_logs/ith_and_best_model/')

    parser.add_argument("--fmri_dir", default='/gpfs/gibbs/pi/krishnaswamy_smita/fmri-meg/fmri/samples_30/')
    parser.add_argument("--meg_dir", default='/gpfs/gibbs/pi/krishnaswamy_smita/fmri-meg/meg/samples_240/')
 
    parser.add_argument("--batch_size", default=1, type=int) 

    # training arguments
    parser.add_argument("--flr", default=0.001, type=float)  # 2e-5
    parser.add_argument("--mlr", default=0.001, type=float)  # 
    parser.add_argument("--n_epochs", default=400, type=int)
    parser.add_argument("--n_steps", default=200000, type=int)
    parser.add_argument("--n_gpus", default=1, type=int) 

    # neural-ode 
    # parser.add_argument("--LO_hidden_size", default=1, type=int) 
    # parser.add_argument("--OF_layer_dim", default=1, type=int) 
    # parser.add_argument("--total_points", default=1, type=int) 
    # parser.add_argument("--obs_points", default=1, type=int) 
    # parser.add_argument("--obsrv_std", default=1, type=int) 
    # parser.add_argument("--rtol", default=1, type=int) 
    # parser.add_argument("--atol", default=1, type=int) 



    # parser.add_argument("--GRU_unit", default=1, type=int) 
    # parser.add_argument("--OR_hidden_size", default=1, type=int) 
    # parser.add_argument("--LO_hidden_size", default=1, type=int) 
    # parser.add_argument("--time_horizon", default=1, type=int)  

 
    parser.add_argument("--device", default='cuda', type=str) 
    cl_args = parser.parse_args()
     
    print("cuda.is_available(): ", torch.cuda.is_available())
    training_fun(cl_args)
