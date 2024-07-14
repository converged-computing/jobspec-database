import numpy as np  # linear algebra
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm, trange
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

seed = 42
BATCH_SIZE = 2**6
NUM_WORKERS = 10
LEARNING_RATE = 5e-5
LR_STEP = 2
LR_FACTOR = 0.2
NUM_EPOCHS = 10
LOG_FREQ = 50
TIME_LIMIT = 100 * 60 * 60
RESIZE = 350
WD = 5e-4
STARTING_EPOCH = 0 # epoch-1
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

os.system(f'mkdir ../model/{sys.argv[1]}')

print(f"RESIZE:{RESIZE}")
print(f"WD: {WD}")
class ImageDataset(Dataset):
    def __init__(self, dataframe, mode):
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        print(f"mode: {mode}, shape: {self.df.shape}")

        transforms_list = [
            transforms.CenterCrop(RESIZE)
        ]

        if self.mode == 'train':
            transforms_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(degrees=(0,360), translate=(0.1, 0.1),
                                            scale=(0.8, 1.2),
                                            resample=Image.BILINEAR)
                ])
            ])

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        ''' Returns: tuple (sample, target) '''
        filename = self.df['Filename'].values[index]

        directory = '../input/Test' if self.mode == 'test' else f'../input/output_combined2_{RESIZE}'
        sample = Image.open(f'./{directory}/gb_12_{filename}')

        assert sample.mode == 'RGB'

        image = self.transforms(sample)

        if self.mode == 'test':
            return image
        else:
            return image, self.df['Drscore'].values[index]

    def __len__(self):
        return self.df.shape[0]

def GAP(predicts: torch.Tensor, targets: torch.Tensor):
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    predicts = predicts.cpu().numpy()
    targets = targets.cpu().numpy()

    res = accuracy_score(targets, predicts)
    return res

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n = 1) :
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch, logging = True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    num_steps = len(train_loader)

    end = time.time()
    lr_str = ''

    for i, (input_, target) in enumerate(train_loader):
        if i >= num_steps:
            break

        output = model(input_.float().to(device))
        loss = criterion(output.flatten(), target.float().to(device))

        predicts = torch.round(output.detach())
        predicts[predicts<0] = 0
        predicts[predicts>4] = 4
        avg_score.update(GAP(predicts, target))

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if logging and i % LOG_FREQ == 0:
            print(f'{STARTING_EPOCH + epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)
            sys.stdout.flush()
        if has_time_run_out():
            break

    print(f' * average GAP on train {avg_score.avg:.4f}')
    return avg_score.avg

def inference(data_loader, model):
    ''' Returns predictions and targets, if any. '''
    model.eval()

    all_predicts, all_targets = [], []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if data_loader.dataset.mode != 'test':
                input_, target = data
            else:
                input_, target = data, None

            output = model(input_.float().to(device))
            predicts = torch.round(output.flatten())
            predicts[predicts<0] = 0
            predicts[predicts>4] = 4
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, targets

def test(test_loader, model):
    predicts, targets = inference(test_loader, model)
    val_loss = F.smooth_l1_loss(predicts.cpu(), targets.float())
    val_loss = val_loss.cpu().numpy().flatten()
    predicts = predicts.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    print(confusion_matrix(targets, predicts))
    print(f"val loss: {val_loss}")
    return cohen_kappa_score(targets, predicts, weights="quadratic")

def train_loop(epochs, train_loader, test_loader, model, criterion, optimizer,
               validate=True):
    train_res = []

    test_res = []
    for epoch in trange(1, epochs + 1):
        print(f"learning rate: {lr_scheduler.get_lr()}")
        start_time = time.time()
        train_acc = train(train_loader, model, criterion, optimizer, epoch, logging=True)
        if has_time_run_out():
            break
        train_res.append(train_acc)
        lr_scheduler.step()

        if validate:
            test_cohen = test(test_loader, model)
            test_res.append(test_cohen)
            print(f"validation score: {test_cohen}")

        torch.save(model.state_dict(), f"../model/{sys.argv[1]}/{sys.argv[1]}_{STARTING_EPOCH+epoch}.ptm")

    return train_res, test_res

def has_time_run_out():
    return time.time() - global_start_time > TIME_LIMIT - 1000

labels = pd.read_csv("../input/training-labels.csv")
train_df, val_df = train_test_split(labels, test_size=0.01,stratify=labels['Drscore'], random_state = seed)
train_dataset = ImageDataset(train_df, mode='train')
val_dataset = ImageDataset(val_df, mode='val')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True,
                          num_workers=NUM_WORKERS)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
# model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
# model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)

model = model.to(device)
model = nn.DataParallel(model)
criterion = nn.SmoothL1Loss()
if len(sys.argv) > 2:
    print(f"loading model: {sys.argv[2]}")
    model.load_state_dict(torch.load(sys.argv[2]))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WD)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP,
                                                   gamma=LR_FACTOR)

global_start_time = time.time()
train_res, test_res = train_loop(NUM_EPOCHS, train_loader, val_loader, model, criterion, optimizer)
sys.stdout.flush()

os.system('sudo shutdown now')
