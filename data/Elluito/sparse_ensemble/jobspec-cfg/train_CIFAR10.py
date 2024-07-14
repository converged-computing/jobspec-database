'''Train CIFAR10 with PyTorch.'''
import copy
from shrinkbench.metrics import flops
import wandb
import omegaconf
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import time
import torch.nn.init as init
import os
import argparse
from alternate_models import *
from pathlib import Path
import pandas as pd
from shrinkbench.metrics.flops import flops
import math
os.environ["LD_LIBRARY_PATH"] = ""

###########################################


# =======================================UTILS===========================================================================
''' Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''


class CosineDecayWithWarmUpScheduler(object):
    def __init__(self, optimizer, step_per_epoch, init_warmup_lr=1e-5, warm_up_steps=1000, max_lr=1e-4, min_lr=1e-6,
                 num_step_down=2000, num_step_up=None,
                 T_mul=1, max_lr_decay=None, gamma=1, min_lr_decay=None, alpha=1):
        self.optimizer = optimizer
        self.step_per_epoch = step_per_epoch
        if warm_up_steps != 0:
            self.warm_up = True
        else:
            self.warm_up = False
        self.init_warmup_lr = init_warmup_lr
        self.warm_up_steps = warm_up_steps
        self.max_lr = max_lr
        if min_lr == 0:
            self.min_lr = 0.1 * max_lr
            self.alpha = 0.1
        else:
            self.min_lr = min_lr
        self.num_step_down = num_step_down
        if num_step_up == None:
            self.num_step_up = num_step_down
        else:
            self.num_step_up = num_step_up
        self.T_mul = T_mul
        if max_lr_decay == None:
            self.gamma = 1
        elif max_lr_decay == 'Half':
            self.gamma = 0.5
        elif max_lr_decay == 'Exp':
            self.gamma = gamma

        if min_lr_decay == None:
            self.alpha = 1
        elif min_lr_decay == 'Half':
            self.alpha = 0.5
        elif min_lr_decay == 'Exp':
            self.alpha = alpha

        self.num_T = 0
        self.iters = 0
        self.lr_list = []

    def update_cycle(self, lr):
        old_min_lr = self.min_lr
        if lr == self.max_lr or (self.num_step_up == 0 and lr == self.min_lr):
            if self.num_T == 0:
                self.warm_up = False
                self.min_lr /= self.alpha
            self.iters = 0
            self.num_T += 1
            self.min_lr *= self.alpha

        if lr == old_min_lr and self.max_lr * self.gamma >= self.min_lr:
            self.max_lr *= self.gamma

    def step(self):
        self.iters += 1
        if self.warm_up:
            lr = self.init_warmup_lr + (self.max_lr - self.init_warmup_lr) / self.warm_up_steps * self.iters
        else:
            T_cur = self.T_mul ** self.num_T
            if self.iters <= self.num_step_down * T_cur:
                lr = self.min_lr + (self.max_lr - self.min_lr) * (
                            1 + math.cos(math.pi * self.iters / (self.num_step_down * T_cur))) / 2
                if lr < self.min_lr:
                    lr = self.min_lr
            elif self.iters > self.num_step_down * T_cur:
                lr = self.min_lr + (self.max_lr - self.min_lr) / (self.num_step_up * T_cur) * (
                            self.iters - self.num_step_down * T_cur)
                if lr > self.max_lr:
                    lr = self.max_lr

        self.update_cycle(lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            self.lr_list.append(lr)
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# #####################################################################################
# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 120
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# =======================================================================================================================


# Training
def train(epoch):
    global best_acc, testloader, device, criterion, trainloader, optimizer, net, use_ffcv, total_flops, batch_flops, record_flops

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == 0:
            print("I just entered the training loop!!!:{}")
        if not use_ffcv:
            inputs, targets = inputs.to(device), targets.to(device)

        # shape_inputs = inputs.shape[0]
        # print("shape_inputs {}".format(shape_inputs))
        # assert shape_inputs[-1] == 360, "Shape inputs: {} expected shape inputs to be {}".format(shape_inputs, 360)
        # break
        if batch_idx == 0:
            print("Inputs device :{}".format(inputs.data.device))
            print("Targets device :{}".format(targets.data.device))
        optimizer.zero_grad()
        if batch_idx == 0:
            t0 = time.time()
        outputs = net(inputs)
        if batch_idx == 0:
            t1 = time.time()
            print("Time for forward pass {}".format(t1 - t0))
        loss = criterion(outputs, targets)
        if batch_idx == 0:
            t0 = time.time()
        loss.backward()
        if record_flops:
            backwardflops = 2 * batch_flops
            total_flops += batch_flops + backwardflops

        if batch_idx == 0:
            t1 = time.time()
            print("Time for backward pass {}".format(t1 - t0))
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(100. * correct / total, correct, total)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # print(
        #     'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if batch_idx % 10 == 0:
            print("Inputs device :{}".format(inputs.data.device))
            print("Outputs device :{}".format(outputs.data.device))

            print(
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total, correct, total


def test(epoch, name="ckpt", save_folder="./checkpoint", args=None):
    global best_acc, testloader, device, criterion
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if batch_idx == 10:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total

    if acc > best_acc and args.save:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'config': args,
        }
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        if os.path.isfile('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc)):
            os.remove('{}/{}_test_acc_{}.pth'.format(save_folder, name, best_acc))
        torch.save(state, '{}/{}_test_acc_{}.pth'.format(save_folder, name, acc))
        best_acc = acc
    return acc


def give_sparse_flops_for_a_batch(model, loader):
    iter_val_loader = iter(loader)
    data, y = next(iter_val_loader)
    _, unit_sparse_flops = flops(model, data)


def get_flops_for_config(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats

    # Data
    print('==> Preparing data..')
    current_directory = Path().cwd()
    data_path = "."
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
        data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "/home/luisaam/Documents/PhD/data/"
    print(data_path)

    batch_size = args.batch_size

    if "32" in args.name:
        batch_size = 32
    if "64" in args.name:
        batch_size = 64

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats_to_use),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "tiny_imagenet":
        from test_imagenet import load_tiny_imagenet
        trainloader, valloader, testloader = load_tiny_imagenet(
            {"traindir": data_path + "/tiny_imagenet_200/train", "valdir": data_path + "/tiny_imagenet_200/val",
             "num_workers": args.num_workers, "batch_size": batch_size})
    if args.dataset == "small_imagenet":
        if args.ffcv:
            from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
            trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train, args.ffcv_val,
                                                                                      batch_size, args.num_workers)
        else:
            from test_imagenet import load_small_imagenet
            trainloader, valloader, testloader = load_small_imagenet(
                {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                 "num_workers": args.num_workers, "batch_size": batch_size})

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet50(num_class=100)
    from torchvision.models import resnet50

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_eatures
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
        if args.type == "pytorch" and args.dataset == "cifar100":
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    if args.model == "vgg19":

        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg_small":

        if args.type == "normal" and args.dataset == "cifar10":
            net = small_VGG_RF("small_vgg", num_classes=10, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_VGG_RF("small_vgg", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)

    total_flops = 0

    solution_name = "{}_{}_{}_rf_level_{}_{}_".format(args.model, args.type, args.dataset, args.RF_level,
                                                      args.name)

    x, y = next(iter(trainloader))

    batch_flops, _ = flops(net, x)

    for e in range(args.epochs):
        for batch in range(len(trainloader)):
            total_flops += batch_flops + 2 * batch_flops

        filepath = "{}/{}_flops.csv".format(args.save_folder, solution_name)

        if Path(filepath).is_file():
            log_dict = {"Epoch": [e], "flops": [total_flops]}

            df = pd.DataFrame(log_dict)
            df.to_csv(filepath, mode="a", header=False, index=False)
        else:
            # Try to read the file to see if it is
            log_dict = {"Epoch": [e], "flops": [total_flops]}
            df = pd.DataFrame(log_dict)
            df.to_csv(filepath, sep=",", index=False)


def get_model_RF(args, RF=None):
    from torchvision.models import resnet50
    assert RF is not None, "RF is None, you need a non null RF level"
    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=RF)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=RF, multiplier=args.width)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=RF, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=RF, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=RF, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_eatures
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
        if args.type == "pytorch" and args.dataset == "cifar100":
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    if args.model == "vgg19":

        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=RF)

        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=RF)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=RF)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=RF)
    if args.model == "resnet_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=RF, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=RF, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=RF, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg_small":

        if args.type == "normal" and args.dataset == "cifar10":
            net = small_VGG_RF("small_vgg", num_classes=10, RF_level=RF)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_VGG_RF("small_vgg", num_classes=100, RF_level=RF)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=RF)

        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=RF)
    return net
def adjust_optim(optimizer, n_iter):
    if n_iter == 1000:
        optimizer.param_groups[0]['betas'] = (0.3, optimizer.param_groups[0]['betas'][1])
    if n_iter > 1000:
        optimizer.param_groups[0]['lr'] *= 0.9999



def iterative_RF_experiments(args):

    print(args)

    global best_acc, testloader, device, criterion, trainloader, optimizer, net, use_ffcv, total_flops, batch_flops, record_flops

    record_flops = args.record_flops

    use_ffcv = args.ffcv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats

    # Data
    print('==> Preparing data..')
    current_directory = Path().cwd()
    data_path = "."
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
        data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "/home/luisaam/Documents/PhD/data/"
    print(data_path)
    batch_size = args.batch_size
    if "32" in args.name:
        batch_size = 32
    if "64" in args.name:
        batch_size = 64

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats_to_use),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "tiny_imagenet":
        from test_imagenet import load_tiny_imagenet
        trainloader, valloader, testloader = load_tiny_imagenet(
            {"traindir": data_path + "/tiny_imagenet_200/train", "valdir": data_path + "/tiny_imagenet_200/val",
             "num_workers": args.num_workers, "batch_size": batch_size})
    if args.dataset == "small_imagenet":
        if args.ffcv:
            from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
            trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train, args.ffcv_val,
                                                                                      batch_size, args.num_workers)
        else:
            from test_imagenet import load_small_imagenet
            trainloader, valloader, testloader = load_small_imagenet(
                {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
            "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution})

    # inputs, y = next(iter(trainloader))
    #
    # print("inputs shape: {}".format(inputs.shape))
    # return 0
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet50(num_class=100)

    net = get_model_RF(args, args.initial_RF)
    # Training
    # # Model
    # print('==> Building model..')
    # # net = VGG('VGG19')
    # # net = ResNet18()
    # # net = PreActResNet18()
    # # net = GoogLeNet()
    # # net = DenseNet121()
    # # net = ResNeXt29_2x64d()
    # # net = MobileNet()
    # # net = MobileNetV2()
    # # net = DPN92()
    # # net = ShuffleNetG2()
    # # net = SENet18()
    # # net = ShuffleNetV2(1)
    # # net = EfficientNetB0()
    # # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = net.to(device)

    net = net.to(device)

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    solution_name = ""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)


    if args.resume:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('{}'.format(args.save_folder)), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load(
        #     '{}/'.format(
        #         args.save_folder))
        checkpoint = torch.load(args.resume_solution)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        for i in range(start_epoch):
            scheduler.step()
        # assert start_epoch == 137, "The start epochs is not 137"
        path = Path(args.resume_solution)
        solution_name = path.stem
        print("solution name: {}".format(solution_name))
    else:
        seed = time.time()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(args.epochs/3)+1)

        solution_name = "{}_{}_{}_{}_rf_level_{}_{}_iterative".format(args.model, args.type, args.dataset, seed,
                                                                      args.RF_level,
                                                                      args.name)
        state = {
            'net': net.state_dict(),
            'acc': 0,
            'epoch': -1,
            "config": args,
        }

        # torch.save(state, '{}/{}_iterative_initial_weights.pth'.format(args.save_folder, solution_name))
    if args.use_wandb:
        os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(omegaconf.DictConfig(vars(args)), resolve=True),
            project="Receptive_Field",
            name=solution_name,
            reinit=True,
            save_code=True,
        )

    total_flops = 0
    x = None
    y = None
    if record_flops:
        x, y = next(iter(trainloader))
        x = x.to(device)

    batch_flops, _ = flops(net, x)

    # lr_list = []
    # for i in range(start_epoch):
    #     for param_group in optimizer.param_groups:
    #         lr_list.append(param_group['lr'])
    #         break
    #     scheduler.step()
    # print("First learning rate:{}".format(lr_list[0]))
    # print("Last learning rate:{}".format(lr_list[-1]))

    epoch_first_change = int(args.ratio_first_change * args.epochs)
    epoch_second_change = int(args.ratio_second_change * args.epochs)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(epoch)
        if epoch == epoch_first_change:
            new_net = get_model_RF(args, args.first_change_RF)
            new_net = new_net.to(device)
            copy_net = copy.deepcopy(net)
            new_net.load_state_dict(copy_net.state_dict())
            net = new_net
            batch_flops, _ = flops(net, x)
        if epoch == epoch_second_change:
            copy_net = copy.deepcopy(net)
            new_net = get_model_RF(args, args.second_change_RF)
            new_net = new_net.to(device)
            new_net.load_state_dict(copy_net.state_dict())
            net = new_net
            batch_flops, _ = flops(net, x)
        t0 = time.time()
        train_acc = train(epoch)
        t1 = time.time()
        print("Epoch time:{}".format(t1 - t0))

        test_acc = test(epoch, solution_name, save_folder=args.save_folder, args=args)

        print("test acc {}".format(test_acc))

        if args.use_wandb:
            # log metrics to wandb
            wandb.log({"Epoch": epoch, "Train Accuracy": train_acc, "Test Accuracy": test_acc})

        if args.record_time:
            filepath = "{}/{}_training_time.csv".format(args.save_folder, solution_name)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "training_time": [t1 - t0]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "training_time": [t1 - t0]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        if args.record_flops:
            filepath = "{}/{}_flops.csv".format(args.save_folder, solution_name)

            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "flops": [total_flops]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "flops": [total_flops]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        if args.record:
            filepath = "{}/{}.csv".format(args.save_folder, solution_name)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "test accuracy": [test_acc], "training accuracy": [train_acc]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "test accuracy": [test_acc], "training accuracy": [train_acc]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        scheduler.step()

    if args.use_wandb:
        wandb.finish()


def main(args):
    print(args)

    global best_acc, testloader, device, criterion, trainloader, optimizer, net, use_ffcv, total_flops, batch_flops, record_flops
    record_flops = args.record_flops
    use_ffcv = args.ffcv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    cifar10_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    cifar100_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    stats_to_use = cifar10_stats if args.dataset == "cifar10" else cifar100_stats

    # Data
    print('==> Preparing data..')
    current_directory = Path().cwd()
    data_path = "."
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data"
    elif "Luis Alfredo" == current_directory.owner() or "Luis Alfredo" in current_directory.__str__():
        data_path = "C:/Users\Luis Alfredo\OneDrive - University of Leeds\PhD\Datasets\CIFAR10"
    elif 'lla98-mtc03' == current_directory.owner() or "lla98-mtc03" in current_directory.__str__():
        data_path = "/jmain02/home/J2AD014/mtc03/lla98-mtc03/datasets"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "/home/luisaam/Documents/PhD/data/"
    print(data_path)
    batch_size = args.batch_size
    if "32" in args.name:
        batch_size = 32
    if "64" in args.name:
        batch_size = 64

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats_to_use),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
    if args.dataset == "tiny_imagenet":
        from test_imagenet import load_tiny_imagenet
        trainloader, valloader, testloader = load_tiny_imagenet(
            {"traindir": data_path + "/tiny_imagenet_200/train", "valdir": data_path + "/tiny_imagenet_200/val",
             "num_workers": args.num_workers, "batch_size": batch_size})
    if args.dataset == "small_imagenet":
        if args.ffcv:
            from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
            trainloader, valloader, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train, args.ffcv_val,
                                                                                      batch_size, args.num_workers)
        else:
            from test_imagenet import load_small_imagenet
            trainloader, valloader, testloader = load_small_imagenet(
                {"traindir": data_path + "/small_imagenet/train", "valdir": data_path + "/small_imagenet/val",
                 "num_workers": args.num_workers, "batch_size": batch_size, "resolution": args.input_resolution})

    # inputs, y = next(iter(trainloader))
    #
    # print("inputs shape: {}".format(inputs.shape))
    # return 0
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet50(num_class=100)
    from torchvision.models import resnet50

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "resnet24":

        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_eatures
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
        if args.type == "pytorch" and args.dataset == "cifar100":
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type,
                                                                                     args.dataset))
    if args.model == "vgg19":

        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet_small":
        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            # net = resnet50()
            # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg_small":

        if args.type == "normal" and args.dataset == "cifar10":
            net = small_VGG_RF("small_vgg", num_classes=10, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "cifar100":
            net = small_VGG_RF("small_vgg", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_VGG_RF("small_vgg", num_classes=200, RF_level=args.RF_level)

    # Training
    # # Model
    # print('==> Building model..')
    # # net = VGG('VGG19')
    # # net = ResNet18()
    # # net = PreActResNet18()
    # # net = GoogLeNet()
    # # net = DenseNet121()
    # # net = ResNeXt29_2x64d()
    # # net = MobileNet()
    # # net = MobileNetV2()
    # # net = DPN92()
    # # net = ShuffleNetG2()
    # # net = SENet18()
    # # net = ShuffleNetV2(1)
    # # net = EfficientNetB0()
    # # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = net.to(device)

    net = net.to(device)

    total_flops = 0
    if record_flops:
        x, y = next(iter(trainloader))
        x = x.to(device)
        batch_flops, _ = flops(net, x)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    solution_name = ""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    if args.resume:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('{}'.format(args.save_folder)), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load(
        #     '{}/'.format(
        #         args.save_folder))
        checkpoint = torch.load(args.resume_solution)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        for i in range(start_epoch):
            scheduler.step()
        # assert start_epoch == 137, "The start epochs is not 137"
        path = Path(args.resume_solution)
        solution_name = path.stem
        print("solution name: {}".format(solution_name))
    else:
        seed = time.time()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        solution_name = "{}_{}_{}_{}_rf_level_{}_{}".format(args.model, args.type, args.dataset, seed, args.RF_level,
                                                            args.name)
        state = {
            'net': net.state_dict(),
            'acc': 0,
            'epoch': -1,
            "config": args,
        }

        torch.save(state, '{}/{}_initial_weights.pth'.format(args.save_folder, solution_name))

    if args.use_wandb:
        os.environ["WANDB_START_METHOD"] = "thread"
        # now = date.datetime.now().strftime("%m:%s")
        wandb.init(
            entity="luis_alfredo",
            config=omegaconf.OmegaConf.to_container(omegaconf.DictConfig(vars(args)), resolve=True),
            project="Receptive_Field",
            name=solution_name,
            reinit=True,
            save_code=True,
        )
    # lr_list = []
    # for i in range(start_epoch):
    #     for param_group in optimizer.param_groups:
    #         lr_list.append(param_group['lr'])
    #         break
    #     scheduler.step()
    # print("First learning rate:{}".format(lr_list[0]))
    # print("Last learning rate:{}".format(lr_list[-1]))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(epoch)
        t0 = time.time()
        train_acc = train(epoch)
        t1 = time.time()
        print("Epoch time:{}".format(t1 - t0))
        test_acc = test(epoch, solution_name, save_folder=args.save_folder, args=args)
        if args.use_wandb:
            # log metrics to wandb
            wandb.log({"Epoch": epoch, "Train Accuracy": train_acc, "Test Accuracy": test_acc})
        if args.record_flops:
            filepath = "{}/{}_flops.csv".format(args.save_folder, solution_name)

            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "flops": [total_flops]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "flops": [total_flops]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        if args.record_time:
            filepath = "{}/{}_training_time.csv".format(args.save_folder, solution_name)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "training_time": [t1 - t0]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "training_time": [t1 - t0]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        if args.record:
            filepath = "{}/{}.csv".format(args.save_folder, solution_name)
            if Path(filepath).is_file():
                log_dict = {"Epoch": [epoch], "test accuracy": [test_acc], "training accuracy": [train_acc]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                # Try to read the file to see if it is
                log_dict = {"Epoch": [epoch], "test accuracy": [test_acc], "training accuracy": [train_acc]}
                df = pd.DataFrame(log_dict)
                df.to_csv(filepath, sep=",", index=False)
        scheduler.step()

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--experiment', default=1, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs to train')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default=7, type=str, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--dataset', default="cifar10", type=str,
                        help='Dataset to use [cifar10,cifar100,tiny_imagenet]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size for trainig')
    parser.add_argument('--model', default="resnet18", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--save_folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location to save the models')
    parser.add_argument('--resume_solution', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Solution from which resume t')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save', default=1, type=int,
                        help='Save best model from trainig')
    parser.add_argument('--input_resolution', default=224, type=int,
                        help='Input Resolution for small ImageNet')
    parser.add_argument('--name', default="", type=str, help='Unique Identifier')
    parser.add_argument('--use_wandb', default=0, type=int, help='Use Weight and Biases')
    parser.add_argument('--width', default=1, type=int, help='Width of the Network')
    parser.add_argument('--record', default=0, type=int, help='To record the training data or not')
    parser.add_argument('--record_time', action='store_true', help="Record the training time")
    parser.add_argument('--record_flops', action='store_true', help="Record the training time")
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')
    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')
    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')
    parser.add_argument('--initial_RF', default=3, type=str, help='Receptive field level')
    parser.add_argument('--first_change_RF', default=5, type=str, help='Second step to reduce receptive field')
    parser.add_argument('--second_change_RF', default=7, type=str, help='Third step to reduce receptive field')
    parser.add_argument('--ratio_first_change', default=0.33, type=float,
                        help='Ratio of training to change RF first time')
    parser.add_argument('--ratio_second_change', default=0.66, type=float,
                        help='Ratio of training to change RF second time')

    args = parser.parse_args()

    try:

        args.RF_level = int(args.RF_level)

    except Exception as e:

        pass

    print(args.resume_solution)
    # return 0
    if args.experiment == 1:
        main(args)
    if args.experiment == 2:
        iterative_RF_experiments(args)

    if args.experiment == 3:
        get_flops_for_config(args)
