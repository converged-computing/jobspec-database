# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets_prep import get_dataset
from models.util import get_flow_model
import lpips
from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from tqdm import tqdm
from torch.autograd.functional import jacobian, jvp


class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.model = model

    def forward(self, t, x_0):
        out = self.model(t, x_0)
        return out
        # return -out[:,:3,:,:] + out[:,3:,:,:]

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def sample_from_model(model, x_0, nfes = [1, 5, 10, 25, 50, 100]):
    t = torch.tensor([1., 0.], device="cuda")
    model_ = Model_(model)

    fake_images = {
    }
    for nfe in nfes:
        fake_image = odeint(model_, x_0, t, method="euler", options = {
                "step_size": 1./nfe,
            })[-1]
        fake_images[nfe] = fake_image
    return fake_images

#%%
def train(rank, gpu, args):

    from EMA import EMA

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    dataset = get_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    model = get_flow_model(args).to(device)
    broadcast_params(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))

    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    #ddp
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=False)

    exp = args.exp
    parent_dir = "./saved_info/flow_matching/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        # load G
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
        # ckpt = torch.load('./saved_info/flow_matching/{}/ckpt/model_{}.pth'.format(args.dataset, 700), map_location=device)
        # print("Finish loading model")
        # model.load_state_dict(ckpt)


    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)

        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            x_0 = x.to(device, non_blocking=True)
            model.zero_grad()
            #sample t
            t = torch.rand((x_0.size(0),) , device=device)
            t = t.view(-1, 1, 1, 1)
            x_1 = torch.randn_like(x_0)
            v_t = (1 - t) * x_1 + t * x_0
            out, jacobian_penalty = jvp(lambda x: model(t.squeeze(), x), v_t, v = torch.randn_like(v_t), create_graph=True)
            penalty = torch.mean(out**2, dim=[0,1,2,3])
            jacobian_penalty = torch.mean(torch.mean(jacobian_penalty**2, dim=[1,2,3]), dim=0)
            loss = F.mse_loss(out, x_1 - x_0) + penalty * 1e-5 + jacobian_penalty * 0.1
            loss.backward()
            optimizer.step()

            if iteration % 50 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, Loss: {}, Penalty: {}, Jab: {}'.format(epoch,iteration, loss.item(), penalty.item(), jacobian_penalty.item()))

        if not args.no_lr_decay:
            scheduler.step()

        if rank == 0:
            rand = torch.randn_like(x_1)[:16]
            fake_samples = sample_from_model(model, rand)
            for nfe in fake_samples.keys():
                torchvision.utils.save_image(fake_samples[nfe], os.path.join(exp_path, 'sample_epoch_{}_{}.png'.format(epoch, nfe)), normalize=True)

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'args': args,
                               'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict()}

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(model.state_dict(), os.path.join(exp_path, 'ema_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)



def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6022'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true',default=False)

    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--num_heads', type=int, default=4,
                            help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1,
                            help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1,
                            help='number of head channels')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,2,2,2),
                            help='channel mult')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)


    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')

    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=800)

    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate g')

    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')


    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')


    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
 
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)