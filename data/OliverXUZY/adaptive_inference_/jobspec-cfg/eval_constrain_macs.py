import os
import argparse

import torch

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, subset_dataset
from libs.model import Worker
from libs.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument(
        '-m', '--macs', type=float, default=1, help='MACs constraint'
    )
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU IDs')
    parser.add_argument('-p', '--load_path', type=str, 
                        default='./log/resnet18_cifar10/train_resnet18_cifar10', 
                        help='path to load model')
    parser.add_argument('--limit', type=int, default=0, help='limit sample to evaluate')
    parser.add_argument('--skip_block', type=int, default=-1, help='how many blocks to skip')

    args = parser.parse_args()
    return args

def main(args):

    if args.load_path:
        ckpt_path = args.load_path
    else:
        os.makedirs('log', exist_ok=True)
        ckpt_path = os.path.join('log', args.name)
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    check_file(cfg_path)
    cfg = load_config(cfg_path)
    print('config loaded from checkpoint folder')

    # configure GPUs
    n_gpus = len(args.gpu.split(','))
    if n_gpus > 1:
        cfg['_parallel'] = True
    set_gpu(args.gpu)

    set_log_path(ckpt_path)
    rng = fix_random_seed(cfg.get('seed', 42))

    ###########################################################################
    """ worker """

    ckpt_name = os.path.join(ckpt_path, 'last.pth')
    check_file(ckpt_name)
    ckpt = torch.load(ckpt_name)
    cfg = ckpt['config']
    worker = Worker(cfg['model'])
    worker.load(ckpt)
    worker.cuda(cfg.get('_parallel'))
    print('worker initialized')

    # print("cfg['model'] branch_vae_cfg = model_cfg: ", cfg['model'].get('branch_vae'))

    ###########################################################################
    """ dataset """

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

    print('val data size: {:d}'.format(len(val_set)))
    if args.limit > 0:
        val_set = subset_dataset(val_set, num_samples = args.limit)
        print('subset val data size: {:d}'.format(len(val_set)))

    ###########################################################################
    """ val """

    metrics_list = ['acc', 'macs']
    metrics = {k: AverageMeter() for k in metrics_list}

    cfg['eval']['max_macs'] = args.macs
    # print(cfg['eval'])  # {'n_branches': 128, 'max_macs': 1, 'batch_size': 128}
    # assert False
    
    for skip_block in range(worker.resnet.n_blocks):
        if args.skip_block >=0:
            skip_block = args.skip_block

        worker.prep_test_branches_constrain_macs(**cfg['eval'], skip_block=skip_block)

        for rx, cx, y in val_loader:
            metrics_dict = worker.eval(rx, cx, y)

            for k in metrics_list:
                if k in metrics_dict.keys():
                    metrics[k].update(metrics_dict[k].item())

        log_str = f'skip {skip_block} block Results:\n'
        for k in metrics_list:
            log_str += '  {:s}\t{:.2f}\n'.format(k, metrics[k].item()*100)
        # print(log_str)
        log(log_str, "eval.txt")

        if args.skip_block >= 0:
            return

    ###########################################################################

if __name__ == '__main__':
    args = parse_args()
    main(args)