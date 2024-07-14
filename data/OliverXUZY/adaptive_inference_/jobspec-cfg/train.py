import os
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.model import Worker
from libs.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="This is my training script.")
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU IDs')
    parser.add_argument('-pf', '--print_freq', type=int, default=1, help='print frequency (x100 itrs)')

    args = parser.parse_args()
    args.print_freq *= 100
    return args


def main(args):
    # set up checkpoint folder
    os.makedirs('log', exist_ok=True)
    ckpt_path = os.path.join('log', args.name)
    ensure_path(ckpt_path)

    # load config
    try:
        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        check_file(cfg_path)
        cfg = load_config(cfg_path)
        print('config loaded from checkpoint folder')
        cfg['_resume'] = True
    except:
        check_file(args.config)
        cfg = load_config(args.config)
        print('config loaded from command line')

    # configure GPUs
    n_gpus = len(args.gpu.split(','))
    if n_gpus > 1:
        cfg['_parallel'] = True
    set_gpu(args.gpu)
    print(f"cfg['_parallel']: {cfg.get('_parallel') or False}")
    # assert False

    set_log_path(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
    rng = fix_random_seed(cfg.get('seed', 42))

    ###########################################################################
    """ worker """

    ep0 = 0
    if cfg.get('_resume'):
        ckpt_name = os.path.join(ckpt_path, 'last.pth')
        try:
            check_file(ckpt_name)
            ckpt = torch.load(ckpt_name)
            ep0, cfg = ckpt['epoch'], ckpt['config']
            worker = Worker(cfg['model'])
            worker.load(ckpt)
        except:
            cfg.pop('_resume')
            ep0 = 0
            worker = Worker(cfg['model'])
    else:
        worker = Worker(cfg['model'])
        yaml.dump(cfg, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

    worker.cuda(cfg.get('_parallel'))
    print('worker initialized, train from epoch {:d}'.format(ep0 + 1))

    ###########################################################################
    """ dataset """
    # print("zhuoyan train: ", cfg['data']['root'])
    train_set = make_dataset(
        dataset=cfg['data']['dataset'],
        root=cfg['data']['root'],
        split=cfg['data']['train_split'], 
        downsample=cfg['data'].get('downsample', False),
    )
    train_loader = make_data_loader(
        train_set, 
        generator=rng,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        is_training=True,
    )

    val_set = make_dataset(
        dataset=cfg['data']['dataset'],
        root=cfg['data']['root'],
        split=cfg['data']['val_split'],
        downsample=cfg['data'].get('downsample', False),
    )
    val_loader = make_data_loader(
        val_set, 
        generator=rng,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        is_training=False,
    )

    itrs_per_epoch = len(train_loader)
    print('train data size: {:d}'.format(len(train_set)))
    print('number of iterations per epoch: {:d}'.format(itrs_per_epoch))

    ###########################################################################
    """ optimizer & scheduler """

    module_list = ['branch_enc', 'content_enc']
    if cfg['model'].get('branch_vae'):
        module_list += ['branch_vae']

    # count number of epochs
    n_epochs = cfg['opt']['epochs']
    if len(cfg['opt']['warmup']) > 0:
        n_epochs += cfg['opt']['warmup_epochs']
    n_itrs = n_epochs * itrs_per_epoch
    
    # build optimizers and schedulers
    optimizers, schedulers = dict(), dict()
    for k in module_list:
        if n_gpus > 1:
            cfg['opt'][k]['lr'] *= n_gpus  # linear scaling rule
        cfg['opt'][k]['epochs'] = n_epochs
        cfg['opt'][k]['itrs_per_epoch'] = itrs_per_epoch
        if k in cfg['opt']['warmup']:
            cfg['opt'][k]['warmup_itrs'] = \
                cfg['opt']['warmup_epochs'] * itrs_per_epoch
        optimizers[k] = make_optimizer(worker, k, cfg['opt'][k])
        schedulers[k] = make_scheduler(optimizers[k], cfg['opt'][k])
        if cfg.get('_resume'):
            optimizers[k].load_state_dict(ckpt['optimizers'][k])
            schedulers[k].load_state_dict(ckpt['schedulers'][k])

    ###########################################################################
    """ train """

    loss_list = ['rank', 'sort', 'bce', 'vae']
    losses = {k: AverageMeter() for k in loss_list}

    metrics_list = ['acc', 'macs']
    metrics = {k: AverageMeter() for k in metrics_list}

    timer = Timer()
    ep_timer= Timer()
    for ep in range(ep0, n_epochs):
        # train for one epoch
        for itr, (rx, cx, y) in enumerate(train_loader, 1):
            loss_dict = worker.train(rx, cx, y, cfg['train'])
            
            if loss_dict is None:
                continue

            global_itr = ep * itrs_per_epoch + itr
            for k in loss_dict.keys():
                if k in loss_list:
                    losses[k].update(loss_dict[k].item())
                    writer.add_scalar(k, losses[k].item(), global_itr)
            writer.flush()
            
            for k in module_list:
                optimizers[k].zero_grad()
            
            loss_dict['total'].backward()
            
            for k in module_list:
                if cfg['opt']['clip_grad_norm'] > 0:
                    nn.utils.clip_grad_norm_(
                        worker.parameters(k), cfg['opt']['clip_grad_norm']
                    )
                optimizers[k].step()
                schedulers[k].step()

            if global_itr == 1 or global_itr % args.print_freq == 0:
                torch.cuda.synchronize()
                t_elapsed = time_str(timer.end())

                log_str = '[{:05d}/{:05d}] '.format(
                    global_itr // args.print_freq, n_itrs // args.print_freq
                )
                for k in loss_dict.keys():
                    if k in loss_list:
                        log_str += '{:s} {:.3f} ({:.3f}) | '.format(
                            k, loss_dict[k].item(), losses[k].item()
                        )
                        losses[k].reset()
                log_str += t_elapsed
                log(log_str, 'log.txt')
                timer.start()

        # save checkpoint
        ckpt = worker.save()
        ckpt['epoch'] = ep + 1
        ckpt['config'] = cfg
        ckpt['optimizers'], ckpt['schedulers'] = dict(), dict()
        for k in module_list:
            ckpt['optimizers'][k] = optimizers[k].state_dict()
            ckpt['schedulers'][k] = schedulers[k].state_dict()
        torch.save(ckpt, os.path.join(ckpt_path, '{:02d}.pth'.format(ep + 1)))
        torch.save(ckpt, os.path.join(ckpt_path, 'last.pth'))

        # validation
        log('\n[ep {:02d} val]'.format(ep + 1), 'log.txt')
        
        worker.prep_test_branches(**cfg['eval'])

        for itr, (rx, cx, y) in enumerate(val_loader, 1):
            metrics_dict = worker.eval(rx, cx, y)

            for k in metrics_list:
                if k in metrics_dict.keys():
                    metrics[k].update(metrics_dict[k].item())

            if itr % args.print_freq == 0:
                torch.cuda.synchronize()
                t_elapsed = time_str(timer.end())

                log_str = '[{:03d}/{:03d}] '.format(
                    itr // args.print_freq, len(val_loader) // args.print_freq
                )
                for k in metrics_list:
                    if k in metrics_dict.keys():
                        log_str += '{:s} {:.3f} ({:.3f}) | '.format(
                            k, metrics_dict[k].item(), metrics[k].item()
                        )
                log_str += t_elapsed
                log(log_str, 'log.txt')

        log_str = 'final: '
        for k in metrics_list:
            log_str += '{:s} {:.3f} | '.format(k, metrics[k].item())
            metrics[k].reset()
        log_str += time_str(timer.end()) + '\n'
        log(log_str, 'log.txt')
        timer.start()

        log(f"total time: {time_str(ep_timer.end())} | {time_str(ep_timer.end()/(ep+1-ep0)*(n_epochs-ep0))}")

    writer.close()
    print('all done!')

    ###########################################################################

if __name__ == '__main__':
    args = parse_args()
    main(args)
