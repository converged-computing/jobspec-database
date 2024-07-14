import sys
sys.path.append('/scratch/rd804/particle_transformer')
#import networks.example_ParticleTransformer as part
import networks.example_ParticleTransformer_finetune as part_finetune
from weaver.utils.dataset import DataConfig
import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger
import torch
import numpy as np
import copy
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
import argparse
from socket import gethostname
from sklearn.metrics import roc_auc_score, roc_curve, auc
import time
import ast

#from weaver.utils.nn.optimizer.lookahead import Lookahead
#import matplotlib.pyplot as plt


# Borrowed from the optimizer in weaver ............
def optim(args, model):
    """
    Optimizer and scheduler.
    :param args:
    :param model:
    :return: opt

    Applies weight decay to all parameters except bias and layer normalization parameters.
    """
    optimizer_options = {k: ast.literal_eval(v) for k, v in args.optimizer_option}
    print('Optimizer options: %s' % str(optimizer_options))

    names_lr_mult = []
    if 'weight_decay' in optimizer_options or 'lr_mult' in optimizer_options:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py#L31
        import re
        decay, no_decay = {}, {}
        names_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or (
                    hasattr(model, 'no_weight_decay') and name in model.no_weight_decay()):
                no_decay[name] = param
                names_no_decay.append(name)
            else:
                decay[name] = param

        decay_1x, no_decay_1x = [], []
        decay_mult, no_decay_mult = [], []
        mult_factor = 1
        if 'lr_mult' in optimizer_options:
            pattern, mult_factor = optimizer_options.pop('lr_mult')
            for name, param in decay.items():
                if re.match(pattern, name):
                    decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    decay_1x.append(param)
            for name, param in no_decay.items():
                if re.match(pattern, name):
                    no_decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    no_decay_1x.append(param)
            assert(len(decay_1x) + len(decay_mult) == len(decay))
            assert(len(no_decay_1x) + len(no_decay_mult) == len(no_decay))
        else:
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
        wd = optimizer_options.pop('weight_decay', 0.)
        parameters = [
            {'params': no_decay_1x, 'weight_decay': 0.},
            {'params': decay_1x, 'weight_decay': wd},
            {'params': no_decay_mult, 'weight_decay': 0., 'lr': args.start_lr * mult_factor},
            {'params': decay_mult, 'weight_decay': wd, 'lr': args.start_lr * mult_factor},
        ]
        print('Parameters excluded from weight decay:\n - %s', '\n - '.join(names_no_decay))
        if len(names_lr_mult):
            print('Parameters with lr multiplied by %s:\n - %s', mult_factor, '\n - '.join(names_lr_mult))
    else:
        parameters = model.parameters()

    if args.optimizer == 'ranger':
        from weaver.utils.nn.optimizer.ranger import Ranger
        opt = Ranger(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adamW':
        opt = torch.optim.AdamW(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'radam':
        opt = torch.optim.RAdam(parameters, lr=args.start_lr, **optimizer_options)


    return opt




parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--save_dir', default='results/top/', type=str)
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'adamW', 'radam', 'ranger'],  # TODO: add more
                    help='optimizer for the training')
parser.add_argument('--optimizer-option', nargs=2, action='append', default=[],
                    help='options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`')
parser.add_argument('--start_lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--use_amp', action='store_true', default=False)
# cuda amp was used by the authors


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def R_X(y_true,y_pred,thr): 
    #to find the threshold that gives TPR of 50%, and the false positive rate at a TPR of 50% 
    fpr_, tpr_, thresholds_ = roc_curve(y_true, y_pred)

    t50 = thresholds_[np.argmin(np.absolute(tpr_-thr))]

    fpr50 = fpr_[thresholds_==t50].item()
    tpr50 = tpr_[thresholds_==t50].item()
    if fpr50>0:
        print('Rejection at tpr ', tpr50, ' is ',1/fpr50)
    
    #print(tpr50)
    return  t50, fpr50, tpr50



def run_epoch(model, dataloader, sampler, optimizer, 
              local_rank, rank, epoch=None, mode = 'train', save_score = False,
              grad_scaler=None):
    if mode == 'train':
        model.train()
        sampler.set_epoch(epoch)
    else:
        model.eval()
    
    if save_score:
        score = []
        label_list = []
    
    running_loss = 0.0
    print(f'grad_scaler: {grad_scaler}')

    for i,(vector, features, mask, labels) in enumerate(dataloader):
        if mode == 'train':
            #for key, opt in optimizer.items():
             #   opt.zero_grad()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                out = model(features.to(local_rank), vector.to(local_rank), mask.to(local_rank))

            if i == 0 and rank == 0:
                print(f'epoch: {epoch} ',' labels: ', labels.shape, 'labels local_rank: ', labels.to(local_rank).shape, ' data: ',
                  ' rank: ', rank, ' local_rank: ', local_rank)
            
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                    out = model(features.to(local_rank), vector.to(local_rank), mask.to(local_rank))

            # check if the output is nan
            if torch.isnan(out).any():
                # count nans
                print(f'Nan in output: {torch.isnan(out[:,0]).sum()}')
                print(f'Nan in labels: {torch.isnan(labels).sum()}')
                print(f'Nan in features: {torch.isnan(features).sum()}')
                print(f'Nan in vector: {torch.isnan(vector).sum()}')
                print(f'Nan in mask: {torch.isnan(mask).sum()}')
                
                #print(f'Nan in output: {out[:1}')
 


       # loss = F.cross_entropy(out, labels.to(local_rank))
        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            loss = torch.nn.CrossEntropyLoss()(out, labels.to(local_rank))
        if save_score:
            score.append(torch.softmax(out, dim=1))
            label_list.append(labels.to(local_rank))


        if rank == 0:
            running_loss += loss.item()
            #loss_list.append(loss.item())
        if mode == 'train':
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
           # for key, opt in optimizer.items():
            #    opt.step()
                

    if rank == 0:
        loss = running_loss / len(dataloader)

    if save_score:
        score = torch.concatenate(score, axis=0)
        label_list = torch.concatenate(label_list, axis=0)

        score = torch.concatenate([score, label_list], axis=1)
        return loss, score
    else:
        return loss

    #return loss, out




def run_multigpu(model, dataloader, sampler, optimizer=None, scheduler=False ,epochs: int = None, \
        world_size: int = 1, rank: int = 1, local_rank: int = 1,
        mode: str = 'train', use_amp: bool = False):
    # Will query the runtime environment for `MASTER_ADDR` and `MASTER_PORT`.
    # Make sure, those are set!
    best_loss = 1000

    if use_amp:
        grad_scaler = torch.cuda.amp.GradScaler()
        print(f'Using AMP for {mode} ................. ')
        
    else:
        grad_scaler = None

    
    #torch.autograd.set_detect_anomaly(True)

    if mode == 'train':
        train_loss_list = []
        val_loss_list = []
        for epoch in range(0, epochs):
            trainloss = run_epoch(model, dataloader['train'], sampler['train'], optimizer, 
                                local_rank, rank, mode=mode, epoch=epoch,
                                grad_scaler=grad_scaler)
            valloss = run_epoch(model, dataloader['val'], sampler['val'], optimizer, 
                                local_rank, rank, mode='val', epoch=epoch,
                                grad_scaler=grad_scaler)

          #  print(f'Epoch: {epoch:02d}, Loss: {trainloss:.4f}')
          #  print(f'Epoch: {epoch:02d}, Loss: {valloss:.4f}')
            dist.barrier()

            if rank == 0:
                print(f'Epoch: {epoch:02d}, train loss: {trainloss:.4f}')
                print(f'Epoch: {epoch:02d}, val loss: {valloss:.4f}')
                train_loss_list.append(trainloss)
                val_loss_list.append(valloss)
                if scheduler:
                    for key, sched in scheduler.items():
                        sched.step(valloss)
                   # scheduler.step(valloss)

                if valloss < best_loss:
                    best_loss = valloss
                    torch.save(model.state_dict(), f'{args.save_dir}/ParT_kin_finetune.pt')
                  #  best_model = copy.deepcopy(model)


            dist.barrier()

       # dist.destroy_process_group()

       # print('loss_list: ', loss_list)
        #loss_list = np.array(loss_list)
        train_loss_list = np.array(train_loss_list)
        val_loss_list = np.array(val_loss_list)

        return train_loss_list, val_loss_list
    else:
        if rank == 0:
            print('Running test')
        dist.barrier()
        loss, out = run_epoch(model, dataloader['test'], sampler['test'], optimizer, 
                              local_rank, rank, mode=mode, save_score=True)
        if rank == 0:
            print('out shape: ', out.shape)

        pred = [torch.zeros_like(out) for _ in range(dist.get_world_size())]
        dist.all_gather(pred, out)
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()

        print('pred shape: ', pred.shape)

        

        dist.barrier()
      #  dist.destroy_process_group()
        return loss, out


if __name__ == '__main__':
    # Get the world size from the WORLD_SIZE variable or directly from SLURM:
    args = parser.parse_args()
    start = time.time()


    rank          = int(os.environ["SLURM_PROCID"])
    world_size    = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    device_name = torch.cuda.get_device_name()

    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
        f" {gpus_per_node} allocated GPUs per node.", flush=True)

    split = ['train','val' ,'test']
    data_features = ['part_4momenta','part_features', 'part_labels']
    raw_data = {}

    for s in split:
        raw_data[s] = {}
        for f in data_features:
            data = np.load(f'data/converted/tops/{s}_{f}.npy')
            #if f=='part_features':
            if f != 'part_labels':
                # clip values between -5 and 5
                data = np.clip(data, -5.0, 5.0)
                # change nan values to 0
               # data = np.nan_to_num(data, nan=0.0)

            raw_data[s][f] = torch.from_numpy(data).float()  

        mask = torch.tensor((raw_data[s]['part_4momenta'].numpy()!=[0.,0.,0.,0.])[...,0]).unsqueeze(-1)
        raw_data[s]['mask'] = torch.swapaxes(mask, 1, 2)

        for f in data_features[:-1]:
            raw_data[s][f] = torch.swapaxes(raw_data[s][f], 1, 2)




       # part_4momenta = torch.tensor(part_4momenta)
       # vector = torch.swapaxes(part_4momenta, 1, 2)
       # features = torch.swapaxes(part_features, 1, 2)
       # mask = torch.swapaxes(mask, 1, 2)
   # print('data: ', _data.shape, ' labels: ', _labels.shape)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    dataset = {}
    sampler = {}
    dataloader = {}
   # dist.init_process_group('nccl', world_size=world_size, rank=rank)

    for s in split:
        #if s == 'train':
        dataset[s] = torch.utils.data.TensorDataset(raw_data[s]['part_4momenta'],
                                                    raw_data[s]['part_features'],
                                                    raw_data[s]['mask'], 
                                                    raw_data[s]['part_labels'])
       # else:
        #    dataset[s] = torch.utils.data.TensorDataset(test_data, test_labels)

        sampler[s] = torch.utils.data.distributed.DistributedSampler(dataset[s], 
                                                                     num_replicas=world_size,
                                                                    rank=rank,
                                                                     shuffle=True if s == 'train' else False)
        dataloader[s] = torch.utils.data.DataLoader(dataset[s], 
                                             sampler=sampler[s], 
                                             batch_size=args.batch_size, 
                                             num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                            pin_memory=True)
 

    data_config_file = 'data/TopLandscape/top_kin.yaml'
    data_config = DataConfig.load(data_config_file, load_observers=False, load_reweight_info=False)
    # count number of parameters

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.use_amp:
        network_options = {'use_amp': True}
        model, model_info = part_finetune.get_model(data_config, **network_options)
    else:
        model, model_info = part_finetune.get_model(data_config)
   # model.to(device)
    #model.mod.requires_grad_(False)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    path_part = 'models/ParT_kin.pt'
    model.load_state_dict(torch.load(path_part),strict=False)
    model = model.to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim(args, model)
   # optimizer = torch.optim.AdamW(model.module.mod.parameters(), lr=3e-4)
   # optimizer2 = torch.optim.AdamW(model.module.fc.parameters(), lr=5e-3, weight_decay=0.01)

  #  sheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=2, verbose=True)
  #  sheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=2, verbose=True)
   # optimizer1 = torch.optim.RAdam(model.module.mod.parameters(), lr=1e-4,
    #                               betas=(0.95, 0.999), eps=1e-05, weight_decay=0.01)
   # optimizer2 = torch.optim.RAdam(model.module.fc.parameters(), lr=5e-3,
    #                               betas=(0.95, 0.999), eps=1e-05, weight_decay=0.01)
   # optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4,
    #                               betas=(0.95, 0.999), eps=1e-05, weight_decay=0.01)
    #lookahead1 = Lookahead(optimizer1, k=6, alpha=0.5)
    #lookahead2 = Lookahead(optimizer2, k=6, alpha=0.5)

    #optimizer = {'part': optimizer1, 'fine_tune': optimizer2}
   # scheduler = {'part': sheduler1, 'fine_tune': sheduler2}
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
   # optimizer_part = torch.optim.Adam(model.mod.parameters(), lr=0.001)
   # optimizer_fine_tune = torch.optim.Adam(model.fc.parameters(), lr=0.001)

   # optimizer = {'part': optimizer_part, 'fine_tune': optimizer_fine_tune}
    if rank == 0:
        print('length of dataloader: ', len(dataloader['train']))
        print('length of dataset: ', len(dataset['train']))
        #print('length ')
        print('length of sampler: ', len(sampler['train']))
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    
    trainloss, valloss = run_multigpu(model, dataloader, sampler, 
                 optimizer=optimizer, scheduler=False,
                 epochs=args.epochs, world_size=world_size, rank=rank, 
                 local_rank=local_rank, mode='train',
                 use_amp=args.use_amp)
    

    
    if rank == 0:
       # torch.save(model.state_dict(), f'{args.save_dir}/ParT_kin_finetune.pt')
       # print('loss: ', loss)
        np.save(f'{args.save_dir}/trainloss.npy', trainloss)
        np.save(f'{args.save_dir}/valloss.npy', valloss)


    print('Finished training, loading best model')
    model.load_state_dict(torch.load(f'{args.save_dir}/ParT_kin_finetune.pt'))
    
    loss, out = run_multigpu(model, dataloader, sampler,
                             world_size=world_size, rank=rank,
                            local_rank=local_rank, mode='test',
                            use_amp=args.use_amp)
    
    if rank == 0:
       # np.save('loss.npy', loss)
        np.save(f'{args.save_dir}/pred.npy', out.detach().cpu().numpy())

        y_true = out[:,-2].detach().cpu().numpy()
        y_pred = out[:,0].detach().cpu().numpy()

        _, _, _ = R_X(y_true, y_pred, 0.3)

        end = time.time()
        print('Time: ', end-start)
    
    dist.destroy_process_group()
    




# for i in range(len(features)//mini_batch):
#     with torch.no_grad():
#         pred.append(model.forward(features[i*mini_batch:(i+1)*mini_batch], vector[i*mini_batch:(i+1)*mini_batch], mask[i*mini_batch:(i+1)*mini_batch]))
#     print(f'Batch {i} done')

#with torch.no_grad():
 #   pred = model.forward(features[:100], vector[:100], mask[:100])
