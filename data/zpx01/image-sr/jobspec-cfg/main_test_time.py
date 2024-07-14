import os
from pathlib import Path
import argparse
import sys
import glob
import numpy as np
import torch
from models.network_swinir import SwinIR as net
import os.path
from os.path import exists
import time
import random
from data_loader import DataLoaderPretrained
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

GPU_DEVICES = 6 # Set to number of GPUs available!


def get_args_parser():
    parser = argparse.ArgumentParser('Test Time Training - Classical SR', add_help=False)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--no_opt', type=bool, default=True, help="If True, then we use an optimizer from scratch, otherwise, we use the provided optimizer.")
    parser.add_argument('--opt_path', type=str, default='/')
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--num_images', default=10, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--test_dir', type=str, help='testset location') 
    parser.add_argument('--output_dir', type=str, help="location for new model checkpoints")
    parser.add_argument('--reset', type=bool, default=True, help='set to False if you do not want to reset optimizer')
    parser.add_argument('--zero_loss', type=bool, default=True, help='set to True if you want the TTT for each image to train till zero loss')
    parser.add_argument('--save_freq', type=int, default=2, help="frequency of saving model checkpoints (saving nth model)")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device")
    return parser

def gpu_info(device):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB\n')

def init_test_time(args, model_path, optimizer_path, device):
        # instantiate model
        model = net(upscale=args.scale, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.2, lr=1e-5)
        param_key_g = 'params'
        model = model.to(device)
        optimizer_to(optimizer, device)

        # load pretrained model
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        if not args.no_opt:
            pretrained_optimizer = torch.load(optimizer_path, map_location='cpu')
            optimizer.load_state_dict(pretrained_optimizer[param_key_g] if param_key_g in pretrained_optimizer.keys() else pretrained_optimizer)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        criterion = torch.nn.L1Loss().cuda()
        return model, pretrained_model, pretrained_optimizer, scheduler, criterion


def main(args):
    # use cuda if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    model_path = args.model_path
    optimizer_path = args.opt_path

    model, pretrained_model, pretrained_optimizer, scheduler, criterion = init_test_time(args, model_path, optimizer_path, device)
    # setting device on GPU if available, else CPU
    print('Using device:\n', device)

    #Additional Info when using cuda
    # gpu_info(device)

    # instantiate model
    param_key_g = 'params'

    # set random seed
    seed = args.seed
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = args.output_dir
    folder = args.test_dir
    best_prec1 = 0
    files = sorted(glob.glob(os.path.join(folder, '*')))[::-1]
    device_index = int(args.device.split(':')[-1])
    files = [f for (i, f) in enumerate(files) if i % GPU_DEVICES == device_index]
    # TTT checkpoint loop for each test image
    for idx, path in enumerate(files):
        print(path)
        img_name = path[45:].replace('.png', '')
        img_name = img_name.replace('/', '')
        if os.path.exists(f'{save_dir}/checkpoint_swinir_{img_name}_last.pth'):
            continue
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255
        data_loader = DataLoaderPretrained(img_lq, sf=args.scale)
        data_loader.generate_pairs(args.num_images)
        dataset = list(zip(data_loader.lr, data_loader.hr))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        if not args.zero_loss:
            for epoch in range(args.epochs):
                adjust_learning_rate(optimizer, epoch)
                prec1, loss = train(args, data_loader, model, optimizer, criterion, epoch, device)
                best_prec1 = max(prec1, best_prec1)
                state_dict = {
                    'epoch': epoch + 1,
                    'arch': 'swinir',
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }
                if (epoch) % args.save_freq == 0:
                    torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_{img_name}_{epoch}.pth')
        else:
            epoch = 0
            loss = float('inf')
            adjust_learning_rate(optimizer, epoch)
            loss_values = []
            best_model = (float('inf'), None)
            while not has_loss_plateaued(loss_values, threshold=0.0001, lookback=12) and epoch < 100:
                prec1, avg_loss, cur_loss = train(args, data_loader, model, optimizer, criterion, epoch, device)
                loss_values.append(cur_loss)
                best_prec1 = max(prec1, best_prec1)
                state_dict = {
                    'epoch': epoch + 1,
                    'arch': 'swinir',
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }
                if cur_loss <= best_model[0]:
                    best_model = (cur_loss, state_dict)
                if (epoch) % args.save_freq == 0 and epoch != 0:
                    state_dict = best_model[1]
                    torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_{img_name}_{epoch}.pth')
                epoch += 1
                scheduler.step(loss)
            state_dict = best_model[1]
            torch.save(state_dict['state_dict'], f'{save_dir}/checkpoint_swinir_{img_name}_last.pth')
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        if args.reset:
            if not args.no_opt:
                optimizer.load_state_dict(pretrained_optimizer[param_key_g] if param_key_g in pretrained_optimizer.keys() else pretrained_optimizer)
            else:
                optimizer = torch.optim.Adam(model.parameters())
                optimizer_to(optimizer, device)
            scheduler = ReduceLROnPlateau(optimizer, 'min')


def optimizer_to(optim, device):
    """
    Move pretrained optimizer to device used by PyTorch
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def train(args, data_loader, model, optimizer, criterion, epoch, device):
    """
    TTT training, updates pretrained model with new augmentations 
    of test image
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    model.train()
    for idx, data in enumerate(data_loader):
        # measure data loading time
        optimizer.zero_grad()
        data_time.update(time.time() - end)

        input, target = data
        input = input.float().to(device)
        target = target.float().to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # record loss
        losses.update(loss.data.item(), input.size(0))

        # do optimizer step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch % 2 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    return top1.avg, losses.avg, losses.val

def adjust_learning_rate(optimizer, epoch):
    """Decay LR dynamically during TTT."""
    lr = 2e-5 * (0.1 ** (epoch // 2))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def has_loss_plateaued(loss_values, threshold=0.0001, lookback=12, num_plateau=6):
    """Stop TTT training if the difference in the loss in the past [lookback] epochs
       is less than the threshold [num_plateau] times.
    """
    if len(loss_values) < lookback:
        return False
    plateau_count = 0
    for i in range(1, lookback+1):
        if len(loss_values) < lookback+1:
            return False
        diff = abs(loss_values[-i] - loss_values[-i-1])
        if diff <= threshold:
            plateau_count += 1
        else:
            plateau_count = 0
        if plateau_count == num_plateau:
            return True
    return False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)