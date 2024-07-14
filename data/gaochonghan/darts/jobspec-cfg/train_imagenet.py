import os
import sys
import numpy as np
import time
import torch
import utils
import logging
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from config import AugmentConfig
from torch.autograd import Variable
from model import NetworkImageNet as Network


config = AugmentConfig()
logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(config.seed)
    cudnn.benchmark = True
    torch.manual_seed(config.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(config.seed)
    logging.info("config = %s", config)
    num_gpus = torch.cuda.device_count()   
    genotype = config.genotype
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------') 
    model = Network(config.init_channels, CLASSES, config.layers, True, genotype)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, 0.1)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
        )
    data_dir = os.path.join(config.data_path, 'imagenet')
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=config.workers)

#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.decay_period, gamma=config.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.epochs))
    best_acc_top1 = 0
    best_acc_top5 = 0
    lr = config.lr
    for epoch in range(config.epochs):
        current_lr = adjust_lr(optimizer, epoch)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and config.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        if num_gpus > 1:
            model.module.drop_path_prob = config.drop_path_prob * epoch / config.epochs
        else:
            model.drop_path_prob = config.drop_path_prob * epoch / config.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('Train_acc: %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer' : optimizer.state_dict(),
            }, is_best, config.path)
        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if config.epochs -  epoch > 5:
        lr = config.lr * (config.epochs - 5 - epoch) / (config.epochs - 5)
    else:
        lr = config.lr * (config.epochs - epoch) / ((config.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        

def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)

        loss_aux = criterion(logits_aux, target)
        loss += config.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % config.print_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs', 
                                    step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % config.print_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main() 
