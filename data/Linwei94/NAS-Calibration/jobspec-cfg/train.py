import wandb
import socket
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset

from models import ResNet18 as ResNet18_CIFAR10, ResNet34 as ResNet34_CIFAR10, ResNet50 as ResNet50_CIFAR10, \
    ResNet101 as ResNet101_CIFAR10, ResNet152 as ResNet152_CIFAR10
from models import ShuffleNetV2 as ShuffleNetV2_CIFAR10
from models import MobileNet as MobileNet_CIFAR10, MobileNetV2 as MobileNetV2_CIFAR10
from models import EfficientNet as EfficientNet_CIFAR10
from models import ResNeXt29_2x64d as ResNeXt29_2x64d_CIFAR10, ResNeXt29_4x64d as ResNeXt29_4x64d_CIFAR10, \
    ResNeXt29_8x64d as ResNeXt29_8x64d_CIFAR10, ResNeXt29_32x4d as ResNeXt29_32x4d_CIFAR10
from models import VGG11 as VGG11_CIFAR10, VGG13 as VGG13_CIFAR10, VGG16 as VGG16_CIFAR10, VGG19 as VGG19_CIFAR10
from models import LeNet as LeNet_CIFAR10

import torch.backends.cudnn as cudnn
from criterion import CrossEntropyMMCE, CrossEntropySoftECE, CrossEntropyLabelSmooth, KLECE, FocalLoss

from model import NetworkCIFAR as Network
from utils.evaluation import test_performance

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--scheduler', type=str, default='darts',
                    help='use darts setting, or "focal" using focal setting as https://arxiv.org/abs/2002.09437')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--auxloss_coef', type=float, default=1, help='coefficient of auxiliary loss')
parser.add_argument('--criterion', type=str, default='ce', help='default cross entropy loss training')
parser.add_argument('--smooth_factor', type=float, default=0.5, help='smooth factor for label smoothing')
parser.add_argument('--focal_gamma', type=float, default=3.0, help='factor for focal loss')

args = parser.parse_args()
if args.criterion == 'focal':
    args.save = './output/retrain-{}-{}-{}-{}'.format(args.arch, args.criterion, args.focal_gamma,
                                                      time.strftime("%Y%m%d-%H%M%S"))
else:
    args.save = './output/retrain-{}-{}-{}-{}'.format(args.arch, args.criterion, args.auxloss_coef,
                                                      time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

config = args
config.hostname = socket.gethostname()
config.name = "retrain"

model_list = {
    'ResNet18': ResNet18_CIFAR10, 'ResNet34': ResNet34_CIFAR10, 'ResNet50': ResNet50_CIFAR10,
    'ResNet101': ResNet101_CIFAR10, 'ResNet152': ResNet152_CIFAR10,
    'ShuffleNetV2': ShuffleNetV2_CIFAR10,
    'MobileNet': MobileNet_CIFAR10, 'MobileNetV2': MobileNetV2_CIFAR10,
    'EfficientNet': EfficientNet_CIFAR10,
    'ResNeXt29_2x64d': ResNeXt29_2x64d_CIFAR10, 'ResNeXt29_4x64d': ResNeXt29_4x64d_CIFAR10,
    'ResNeXt29_8x64d': ResNeXt29_8x64d_CIFAR10,
    'ResNeXt29_32x4d': ResNeXt29_32x4d_CIFAR10,
    'VGG11': VGG11_CIFAR10, 'VGG13': VGG13_CIFAR10, 'VGG16': VGG16_CIFAR10, 'VGG19': VGG19_CIFAR10,
    'LeNet': LeNet_CIFAR10,
}

wandb.init(project="NAS Calibration", entity="linweitao", config=config)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    if args.arch in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ShuffleNetV2',
                     'MobileNet', 'MobileNetV2', 'EfficientNet', 'ResNeXt29_2x64d', 'ResNeXt29_4x64d',
                     'ResNeXt29_8x64d', 'ResNeXt29_32x4d', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'LeNet']:

        model = model_list[args.arch]()
        args.is_searched_arch = False
        wandb.config.is_searched_arch = False
    else:
        args.is_searched_arch = True
        wandb.config.is_searched_arch = True
        genotype = eval("genotypes.%s" % args.arch)
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    if args.parallel:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    wandb.config.model_size = utils.count_parameters_in_MB(model)
    wandb.watch(model)

    criterion_dict = {
        'ce': nn.CrossEntropyLoss(),
        'softece': CrossEntropySoftECE(CIFAR_CLASSES, args.auxloss_coef),
        'mmce': CrossEntropyMMCE(CIFAR_CLASSES, args.auxloss_coef),
        'ls': CrossEntropyLabelSmooth(CIFAR_CLASSES, args.smooth_factor),
        'klece': KLECE(CIFAR_CLASSES, args.auxloss_coef),
        'focal': FocalLoss(args.focal_gamma)
    }

    criterion = criterion_dict[args.criterion]
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, test_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if args.scheduler == "focal":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        test_acc, test_obj = infer(test_queue, model, criterion)
        logging.info('test_acc %f', test_acc)

        ece, adaece, cece, nll = test_performance(test_queue=test_queue, model=model)
        logging.info('ece %f, adaece %f, cece %f, nll %f', ece, adaece, cece, nll)
        wandb.log({
            "current epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "ece": ece
        })

        scheduler.step()
        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        if args.is_searched_arch:
            logits, logits_aux = model(input)
        else:
            logits = model(input)

        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()

            if args.is_searched_arch:
                logits, logits_aux = model(input)
            else:
                logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
