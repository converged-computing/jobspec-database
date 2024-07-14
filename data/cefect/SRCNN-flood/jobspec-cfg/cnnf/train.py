
#===============================================================================
# IMPORTS------
#===============================================================================
import argparse, pickle, psutil, logging, csv, time
import os
import copy
from datetime import datetime

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import numpy as np

from cnnf.models import SRCNN, init_weights 
from cnnf.datasets import h5Dataset as TrainDataset
#from cnnf.datasets import RasterioDataset as EvalDataset
#from cnnf.datasets import transform_eval
from cnnf.hp import today_str, dstr, init_logger

from definitions import wrk_dir, job_log_file



def train_model(input_data_fp,
                eval_data_fp=None,
                out_dir=None, 
                
                weights_file=None,
                #scale=3,
                lr=1e-4,
                batch_size=16,
                kernel_size=13,
                
                
                num_epochs=400,
                num_workers=8,
                seed=123,
                device=None,
                
                log=None,
                tag='v01',
                hpc=False,
                
 
 
                ):
    
    """train the SCRNN against a dataset with labels
    
    Params
    --------
    input_data_fp: str
    
    eval_data_fp: str
    
    out_dir: str
    
 
    weights_file: str
        optional starter weights file
        
    lr: float
        optimizer's learning rate
        
    batch_size: int
        size to batch data into
        
    num_epochs: int
        number of training epochs
        
    num_workers: int
        number of works for loading data in parallel
        
        
    seed: int
        random seed for initilizeing paramaters
        
    hpc: bool
        flag to control some hpc-specific logging
        
    """
    #===========================================================================
    # setup
    #===========================================================================
    start=datetime.now()
    #configure outputs
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', f'train', tag)
 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if log is None: 
        log = init_logger(name='train',fp=os.path.join(out_dir, f'train_{os.getpid()}_{today_str}.log'))
 
    
    #set device
    cudnn.benchmark = True
    
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    log.info(f'running on device: {device}')
    torch.manual_seed(seed)
    
    #===========================================================================
    # initialize-----
    #===========================================================================
    log.info('init model')
    model = SRCNN(log=log, kernel_size_i=kernel_size).to(device)
    
    if weights_file:
        log.info(f'loading model weights from file {os.path.basename(weights_file)}')
        model.load_state_dict(torch.load(weights_file))
    else:
        #use random weights
        pass
        #couldnt get uniform weights to work
        #log.info(f'applying uniform weights')
        #model.apply(init_weights)
    
    criterion = nn.MSELoss() #used in the training loop
    eval_criterion = nn.MSELoss() #used in the evaluation loop
    
    # first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments
    optimizer = optim.Adam([
            {'params':model.conv1.parameters()}, 
            {'params':model.conv2.parameters()}, 
            {'params':model.conv3.parameters(), 'lr':lr * 0.1}], 
        lr=lr)
    #===========================================================================
    # data
    #===========================================================================
    log.info(f'init data')
    
    train_dataset = TrainDataset(input_data_fp, log=log)
    """
    train_dataset.print_atts()
    """
    train_dataloader = DataLoader(dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True)
 

    #eval data
    if eval_data_fp:
        eval_dataset = TrainDataset(eval_data_fp, log=log, 
                                    transform_all=None,
                                    )
        
        eval_dataloader = DataLoader(dataset=eval_dataset, 
                                     batch_size=1,                                     
                                     )
        
    #check data loaders
    if __debug__:
        log.debug(f'checking data loaders')
        for _ in train_dataloader:
            pass
        
        if eval_data_fp:
            for _ in eval_dataloader:
                pass
    
    batch_group_cnt = (len(train_dataset) - len(train_dataset) % batch_size)
    assert batch_group_cnt>0, f'bad batch size?'
    
    #===========================================================================
    # train epoch loop-------
    #===========================================================================
    log.info(f'start training {num_epochs} loops\n-------------------')
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 9e9
    backprops = 0
    
    #meta_lib={'meta':{'input_data_fp':input_data_fp, 'eval_data_fp':eval_data_fp}}
    res_d=dict()
    meta_lib=dict()
    
    for epoch in range(num_epochs):
        meta_d={'start':datetime.now()}
        model.train()
        
        #epoch_losses = AverageMeter()
        
        with tqdm(total=batch_group_cnt, disable=(__debug__ or hpc)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))
            
            #loop on batched tensor data
            eval_d = dict()
            for i, (target_tensor, input_tensor) in enumerate(train_dataloader):
                
                log.debug(f'{i}    target: {target_tensor.shape} input: {input_tensor.shape}'+\
                          f'mem: {psutil.virtual_memory()[3] / 1000000000}')
                
                inputs = input_tensor.to(device)
                targets = target_tensor.to(device)                
 
                #compute the prediction on teh model
                preds = model(inputs) 
                
                #compute the loss, comparing the first input to the target
                loss = criterion(preds.squeeze(1), targets)                
                

                eval_d[i] = loss.item() #/len(inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

                t.set_postfix(loss='{:.6f}'.format(loss.item()))
                t.update(len(inputs))
                backprops += 1
        
        
        #=======================================================================
        # write these epoch parameters
        #=======================================================================
        model_fp = os.path.join(out_dir, 'modelState_epoch_{}.pth'.format(epoch))
        log.info(f'saving model state to \n    {model_fp}')
        torch.save(model.state_dict(), model_fp)
        
        #update meta
        meta_d.update({'backprops':backprops, 'iters':i+1, 
                       'loss_mean':np.array(list(eval_d.values())).mean(), #dont really care about this
                       'model_fp':model_fp, 'model_fn':os.path.basename(model_fp)})
        
        #=======================================================================
        # evaluate this epoch-------
        #=======================================================================
        if eval_data_fp:
            log.debug(f'evaluating against {os.path.basename(eval_data_fp)}')
            model.eval()
            #epoch_psnr = AverageMeter()
            
            #loop through each patch in the evaluation dataset (batch_size=1)
            eval_d=dict()
            with torch.no_grad():
                for i, (target_tensor, input_tensor) in enumerate(eval_dataloader): 
                    
                    #prep the data
                    assert len(input_tensor)==1, f'only setup for single batching here'

                    inputs = input_tensor.to(device)
                    targets = target_tensor.to(device)
                    
                    #compute the prediction (cap values to zero and 1)                                        
                    preds = model(inputs)
                        
                    eval_d[i] = eval_criterion(preds.squeeze(1), targets).item()
                    #epoch_psnr.update(eval_criterion(preds.squeeze(1), targets), n=len(inputs))
            
            #stat on all patches
            loss_mean = np.array(list(eval_d.values())).mean()
            
            log.info('eval loss: {:.2f}, train loss: {:.2f} (backprops={})'.format(loss_mean, loss.item(), backprops))
            
            #wrap eval
            if loss_mean < best_loss:
                best_epoch, best_loss = epoch, loss_mean
                best_weights = copy.deepcopy(model.state_dict())
                meta_d['best'] = True
            else:
                meta_d['best'] = False
                
            meta_d['eval_loss_mean'] = loss_mean
                     
        #=======================================================================
        # wrap epoch
        #=======================================================================
        meta_d['tdelta'] = datetime.now() - meta_d['start']
        meta_lib[epoch] = meta_d
    
    #===========================================================================
    # wrap------- 
    #===========================================================================
    log.info(f'\n\nWRAP\n---------------------')
    if eval_data_fp:
        log.info('best epoch: {}, best_loss: {:.4f}'.format(best_epoch, best_loss))
        res_d['eval_data_fp'] = os.path.join(out_dir, f'modelState_bestEpoch_{today_str}.pth')
        torch.save(best_weights, res_d['eval_data_fp'])
        log.info(f'saved to \n    %s'%res_d['eval_data_fp'])
    
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        #'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
        }
        
    #===========================================================================
    # meta
    #===========================================================================
    try:
        import pandas as pd
        meta_df = pd.DataFrame.from_dict(meta_lib).T    
        res_d['meta_fp'] = os.path.join(out_dir, f'meta_{today_str}.csv')
        meta_df.to_csv(res_d['meta_fp'])
        log.info(f'wrote meta {str(meta_df.shape)} to \n   %s'%res_d['meta_fp'])
    except Exception as e:
        log.error(f'failed to write meta csv w/ \n    {e}')
    
    #as a pickle 
    try:
        with open(os.path.join(out_dir, f'meta_{today_str}.pkl'), 'wb') as f:
            pickle.dump(meta_lib, f)
    except Exception as e:
        log.error(f'failed to write meta pickle w/ \n    {e}')
        
    

        
    #to job log
    try:
        hmeta_d = {'tag':tag,
                   'now':datetime.now().isoformat(), #convert this to a string
            'input_data_fp':input_data_fp,'eval_data_fp':eval_data_fp, 'out_dir':out_dir,
            'lr':lr, 'batch_size':batch_size, 'num_epochs':num_epochs, 'seed':seed,
            'best_epoch':best_epoch, 'best_loss':best_loss, 'kernel_size':kernel_size, 'num_workers':num_workers,           
            }
        
        hmeta_d.update(meta_d)
        
        #add hmeta_d as a line to the csv file
        #the the job_log_file is write locked, wait 10 seconds and try again
        while True:
            try:
                # check if file exists, if not create it
                if not os.path.exists(job_log_file):
                    with open(job_log_file, 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(hmeta_d.keys())  # write keys as header
        
                with open(job_log_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(hmeta_d.values())
                break
            except IOError:
                time.sleep(10)
                
        log.info(f'wrote hyper metadat to \n    {job_log_file}')
    except Exception as e:
        log.error(f'failed to write to job log w/ \n    {e}')
 
        
    #===========================================================================
    # wrap
    #===========================================================================    
    log.info(f'finished {num_epochs} epochs on \n    {dstr(res_d)}\n    {meta_d}\n    {out_dir}')
    
    return res_d






if __name__ == '__main__':
    #set the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-fp', type=str, required=True)
    parser.add_argument('--eval-data-fp', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=False, default=None)
    parser.add_argument('--weights-file', type=str, required=False, default=None)
    #parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='data batches')
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoading subprocesses to use for data loading')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--tag', type=str, default='v01')
    parser.add_argument('--hpc', type=bool, default=False)
    parser.add_argument('--kernel-size', type=int, default=13)
    args = parser.parse_args()
    
    train_model(args.input_data_fp, args.eval_data_fp, out_dir=args.out_dir, 
                #scale=args.scale,
                lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, num_workers=args.num_workers,
                seed=args.seed, kernel_size=args.kernel_size,
                tag=args.tag)
    
 
    
    
    
    
    
    
    
    
    
    
    
