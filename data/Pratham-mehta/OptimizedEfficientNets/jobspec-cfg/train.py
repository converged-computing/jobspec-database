# ImageNet Efficientnet Pytorch Training
import argparse
import os
import random
import shutil
import time
import warnings
import PIL
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import csv
import torchvision
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from torch.profiler import profile, record_function, ProfilerActivity


parser = argparse.ArgumentParser(description='Pytorch EfficientNet CIFAR-100 Training(finetuning)')
# parser.add_argument('--data_path',default='/vast/sd5023/ImageNet/imagenet',type = str, help='Path to Dataset')
parser.add_argument('--arch', default='efficientnet_b0',type=str, choices=['efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7'])
parser.add_argument('--workers',default=4, type=int, help='number of dataloading workers')
parser.add_argument('--epochs', default=50, type=int,help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='mannual epch number(useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size, this is the total batch size of all GPUs on the current node when using DataParallel or DistributedDataParallel')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--image_size', default=224, type=int, help='image size')
parser.add_argument('--world-size', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--resume', default='',type=str, help='path to latest checkpoint')
parser.add_argument('--dataset', type=str, help=['cifar', 'food'])
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use Pytorch for either single node or multi node dataparallel training')
parser.add_argument('--profile', default= False, type=bool, help='[True, False]')
best_acc1 = 0

def main():
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(42)
        # cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down training considerably! May see unexpected behavior when restarting from checkpoints')
        
    if args.gpu is not None:
        warnings.warn('Have chosen a specific GPU. Will completely disable data parallelism')
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('Number of GPUs used: ', ngpus_per_node)
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        #simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank need to be the global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
     
    if args.dataset == 'cifar':
        num_classes = 100
    elif args.dataset == 'food':
        num_classes = 101
        
    
    if args.arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        model.classifier[1] = nn.Linear(1280, num_classes)
            
    elif args.arch == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='DEFAULT')
        model.classifier[1] = nn.Linear(1280, num_classes) 

    elif args.arch == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(1408, num_classes) 
    elif args.arch == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='DEFAULT')
        model.classifier[1] = nn.Linear(1536, num_classes)
    elif args.arch == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='DEFAULT')
        model.classifier[1] = nn.Linear(1792, num_classes) 
    elif args.arch == 'efficientnet_b5':
        model = models.efficientnet_b5(weights='DEFAULT')
        model.classifier[1] = nn.Linear(2048, num_classes)
    elif args.arch == 'efficientnet_b6':
        model = models.efficientnet_b6(weights='DEFAULT')
        model.classifier[1] = nn.Linear(2304, num_classes) 
    elif args.arch == 'efficientnet_b7':
        model = models.efficientnet_b7(weights='DEFAULT')
        model.classifier[1] = nn.Linear(2560, num_classes)
        
    #Freeze the pre-trained layers
    
    
    # print(model.parameters)
    # Modify the classifier for CIFAR-100 (100 classes)
    # Modify classifier
    # model.classifier[1] = nn.Linear(1280, 100) 
        # Freeze feature parameters
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Classifier parameters   
    for param in model.classifier.parameters():
        param.requires_grad = True
    
        
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise, DistributedDataParallel will use all availaible devices
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            #When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size/ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all availabile GPUs
        print("Using DataParallel")
        model = torch.nn.DataParallel(model).cuda()
        
    
    # Define loss function(criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['epoch']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> Loader checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True
    
    # DataLoading code
    # traindir = os.path.join(args.data_path, 'train')
    # valdir = os.path.join(args.data_path, 'val')
    
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    
    if args.arch == 'efficientnet_b0':
        image_size = 224
    elif args.arch == 'efficientnet_b1':
        image_size = 224
    elif args.arch == 'efficientnet_b2':
        image_size = 260
    elif args.arch == 'efficientnet_b3':
        image_size = 300
    elif args.arch == 'efficientnet_b4':
        image_size = 380
    elif args.arch == 'efficientnet_b5':
        image_size = 456
    elif args.arch == 'efficientnet_b6':
        image_size = 528
    elif args.arch == 'efficientnet_b7':
        image_size = 600
        
    transform_dataset = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    if args.dataset == 'cifar':
        cifar100_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform = transform_dataset)
        train_size = int(0.8 * len(cifar100_dataset))
        val_size = len(cifar100_dataset) - train_size
        train_dataset, val_dataset = random_split(cifar100_dataset, [train_size, val_size])
    elif args.dataset == 'food':
        
        # Create the Food101 dataset object.
        food101_dataset = torchvision.datasets.Food101(root='./data', download=True, transform=transform_dataset)
        
        # Define the length of the training dataset (e.g., 70% of the entire dataset).
        train_length = int(len(food101_dataset) * 0.7)
        
        # The rest will be used for the validation dataset.
        val_length = len(food101_dataset) - train_length
        
        # Use the random_split function to split the Food101 dataset into training and validation datasets.
        train_dataset, val_dataset = random_split(food101_dataset, [train_length, val_length])
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=(train_sampler is None),
                                              num_workers=args.workers, pin_memory=True,
                                              sampler=train_sampler)

    
    # print('Using image size: ', image_size)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,shuffle=False,
                                            num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        res = validate(val_loader, model, criterion, args)
        filen = 'res' + args.arch + '.txt'
        with open(filen, 'w') as f:
            print(res, file=f)
        return
    # Integration of Profiler
    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("main_worker"):
                # Call the modified training function
                train_with_profiler(train_loader,val_loader, model, criterion, optimizer, args)
    
        # Print the profiling results
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print("Training begun")
        
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)
            
            # train for one epoch
            print("For epoch {}".format(epoch))
            train(train_loader, model, criterion, optimizer, epoch, args)
            
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, epoch,args)
            
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
    
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best,filename='/scratch/sd5023/HPML/Course_Project/save_path/checkpoint.pth.tar')
            
def train_with_profiler(train_loader, val_loader,model, criterion, optimizer, args):
    global best_acc1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        print("For epoch {}".format(epoch))
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='/scratch/sd5023/HPML/Course_Project/save_path/profiler/checkpoint.pth.tar')
            
            
            
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    # Open a CSV file to save the metrics
    num_gpus_str = str(torch.cuda.device_count())
    filename = "training_metrics_" + args.arch + '_' + num_gpus_str +  "_profiler_" + args.dataset + ".csv"
    with open(filename, 'a', newline='') as csvfile:
        metric_writer = csv.writer(csvfile)
        if epoch == 0 and args.start_epoch == 0:
            # Write the header only once, at the beginning of the first epoch
            metric_writer.writerow(['Epoch', 'Batch', 'Data Time', 'Loss', 'Batch Time', 'Acc@1', 'Acc@5'])
            
            
        # Define the main training function
        # def train_with_profiler(train_loader, model, criterion, optimizer, epoch, args):
        #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #         with record_function("train"):
        #             train(train_loader, model, criterion, optimizer, epoch, args)

        #     # Print the profiling results
        #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # # Call the modified training function
        # train_with_profiler(train_loader, model, criterion, optimizer, epoch, args)


        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # Explicitly free unused GPU memory
            torch.cuda.empty_cache()
            # Save metrics to CSV
            metric_writer.writerow([epoch, i, data_time.val, losses.val, batch_time.val, top1.val, top5.val])

            if i % args.print_freq == 0:
                progress.print(i)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    num_gpus_str = str(torch.cuda.device_count())
    filename = 'validation_metrics_' + args.arch +'_'+ num_gpus_str + '_profiler_' + args.dataset +'.csv'

    with torch.no_grad():
        end = time.time()
        with open(filename, 'a', newline='') as csvfile:
            metric_writer = csv.writer(csvfile)
            if args.start_epoch == 0:
                # Write the header only once, at the beginning of the first epoch
                metric_writer.writerow(['Epoch', 'Batch', 'Loss', 'Batch Time', 'Acc@1', 'Acc@5'])

            for i, (images, target) in enumerate(val_loader):
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Save metrics to CSV
                metric_writer.writerow([epoch, i, losses.val, batch_time.val, top1.val, top5.val])

                if i % args.print_freq == 0:
                    progress.print(i)

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/scratch/sd5023/HPML/Course_Project/save_path/model_best_efficientnet_b0.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))#(0.1 ** (epoch // 30))#
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
        
    
    
    
    
    
