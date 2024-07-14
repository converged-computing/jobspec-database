'''Train CIFAR10 with PyTorch.'''
from ast import arg
from pyexpat import model
from re import A
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os
import argparse

from torchsummary import summary
from models import *
from utils import *

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default='ResNet', type=str, help='model to use, choose from: ResNet, ResNetDeep, ResNetWide')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer to use, choose from: SGD, Adam...')
parser.add_argument('--scheduler', default='CosineAnnealingLR', type=str, help='scheduler to use, choose from: CosineAnnealingLR, StepLR...')
parser.add_argument('--N', default=3, type=int, help='number of residual layers in ResNet')
parser.add_argument('--B', default=2, type=int, help='number of blocks in each residual layer')
parser.add_argument('--data_augmentation', default=False, type=bool, help='whether to add extra data augmentation')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
augmentation_flag = args.data_augmentation
# Data
print('==> Preparing data..')
if augmentation_flag:
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
print("train data shape: ", trainset.data.shape)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
print("test data shape: ", testset.data.shape)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
kaggle_testset = KaggleDataset('cifar_test_nolabels.pkl', 'data.csv',transform_test)
print("kaggle test data shape: ", kaggle_testset.data.shape)
kaggle_testloader = torch.utils.data.DataLoader(
    kaggle_testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# parameters
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
max_parameters = 5000000
N = args.N
B = args.B


# Model
model_name = args.model
print('==> Building model..')
if model_name == 'ResNet':
    net = ResNetWide0()
elif model_name == 'ResNetWide':
    net = ResNetWide()
elif model_name == 'ResNetDeep':
    net = ResNetDeep()
else:
    print('Model not available')
    exit()
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# print number of parameters of the model
print("total number of parameters: ")# , summary(net, (3, 32, 32)))
model_status = summary(net, (3, 32, 32),verbose=0)
model_status_str = str(model_status)
number_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
if(number_parameters>max_parameters):
    print(f'Number of parameters are {number_parameters}, exceeds the limit of {max_parameters}')
    exit()
# save config
saved_path = f'./new_saved_results/{model_name}_batch_size_{batch_size}_epochs_{start_epoch+epochs-1}_{args.scheduler}_{args.optimizer}'
if augmentation_flag:
    saved_path += '_data_augmentation'
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'{saved_path}/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer
if args.optimizer == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(net.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
elif args.optimizer == 'Adagrad':
    optimizer = torch.optim.Adagrad(net.parameters(), lr=args.lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
elif args.optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
elif args.optimizer == 'SparseAdam':
    optimizer = torch.optim.SparseAdam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
elif args.optimizer == 'Adamax':
    optimizer = torch.optim.Adamax(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
elif args.optimizer == 'ASGD':
    optimizer = torch.optim.ASGD(net.parameters(), lr=args.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
elif args.optimizer == 'LBFGS':
    optimizer = torch.optim.LBFGS(net.parameters(), lr=args.lr, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
elif args.optimizer == 'RMSprop':
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
elif args.optimizer == 'Rprop':
    optimizer = torch.optim.Rprop(net.parameters(), lr=args.lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))  
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
# scheduler
if args.scheduler == 'LambdaLR':
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
elif args.scheduler == 'MultiplicativeLR':
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
elif args.scheduler == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
elif args.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
elif args.scheduler == 'ConstantLR':
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
elif args.scheduler == 'LinearLR':
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_lr=0.1, end_lr=0.0001, total_steps=200)
elif args.scheduler == 'ExponentialLR':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
elif args.scheduler == 'PolynomialLR':
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_iter=200, power=0.9)
elif args.scheduler == 'ChainedScheduler':
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1), torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)])
elif args.scheduler == 'SequentialLR':
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, milestones=[30, 60], gamma=0.1)
elif args.scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
elif args.scheduler == 'CyclicLR':
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
elif args.scheduler == 'OneCycleLR':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=None, epochs=200, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)
elif args.scheduler == 'CosineAnnealingWarmRestarts':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=0, last_epoch=-1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


config = {
    'model_name': model_name,
    'batch_size': batch_size,
    'lr': lr,
    'epochs': start_epoch+epochs-1,
    'optimizer': args.optimizer,
    'scheduler': args.scheduler,
    # 'N': N,
    # 'B': B,
    'model': str(net),
    'summary': model_status_str,
    'total_parameters': number_parameters,
    'scheduler': args.scheduler,
    'optimizer': args.optimizer,
    'augmentation': augmentation_flag
}
try:
    with open(f'{saved_path}/config.json', 'w') as f:
        json.dump(config, f)
        print(f'saved config to {saved_path}')
except (TypeError, OverflowError):
    with open(f'{saved_path}/config.json', 'w') as f:
        config_str = str(config)
        config_str = config_str[config_str.index('{'):]
        json.dump(json.loads(config_str),f)
        print(f'convert from string and saved config to {saved_path}')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total


def test(epoch,dataloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir(f'checkpoint'):
        #     os.mkdir('checkpoint')
        torch.save(state, f'{saved_path}/ckpt.pth')
        best_acc = acc
    return test_loss/(batch_idx+1), correct/total, best_acc

train_losses = []
train_accs = []
test_losses = []
test_accs = []
kaggle_losses = []
kaggle_accs = []
for epoch in range(start_epoch, start_epoch+epochs):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc, test_best_acc = test(epoch,testloader)
    kaggle_loss, kaggle_acc, kaggle_best_acc = test(epoch,kaggle_testloader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    kaggle_losses.append(kaggle_loss)
    kaggle_accs.append(kaggle_acc)
    scheduler.step()
    if epoch == 1 or epoch % 10 == 0:
        if not os.path.exists(f'{saved_path}/epoch_{epoch}'):
            os.makedirs(f'{saved_path}/epoch_{epoch}')
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        ax[0].plot(train_losses)
        ax[1].plot(train_accs)
        ax[0].set_title(f'Train Loss for epoch {epoch}')
        ax[1].set_title(f'Train Accuracy for epoch {epoch}')
        plt.savefig(f'{saved_path}/epoch_{epoch}/train_results.png')
        plt.close(fig)
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        ax[0].plot(test_losses)
        ax[1].plot(test_accs)
        ax[0].set_title(f'Test Loss for epoch {epoch}')
        ax[1].set_title(f'Test Accuracy for epoch {epoch}')
        plt.savefig(f'{saved_path}/epoch_{epoch}/test_results.png')
        plt.close(fig)
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        ax[0].plot(kaggle_losses)
        ax[1].plot(kaggle_accs)
        ax[0].set_title(f'Kaggle Loss for epoch {epoch}')
        ax[1].set_title(f'Kaggle Accuracy for epoch {epoch}')
        plt.savefig(f'{saved_path}/epoch_{epoch}/kaggle_results.png')
        plt.close(fig)
    
    config.update({
        "best_acc": test_best_acc,
        "best_kaggle_acc": kaggle_best_acc
    })
    try:
        with open(f'{saved_path}/config.json', 'w') as f:
            json.dump(config, f)
            print(f'saved config to {saved_path}')
    except (TypeError, OverflowError):
        with open(f'{saved_path}/config.json', 'w') as f:
            config_str = str(config)
            config_str = config_str[config_str.index('{'):]
            json.dump(json.loads(config_str),f)
            print(f'convert from string and saved config to {saved_path}')
