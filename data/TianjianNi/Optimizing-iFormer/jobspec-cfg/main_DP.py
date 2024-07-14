import os
import argparse
from time import perf_counter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.parallel import DataParallel as DP

from dataset import Customized_CIFAR10_Dataset
from model import iformer_small


def train(args):
    batch_size = args.batch_size
    num_workers = args.num_workers
    data_path = args.data_path
    epochs = args.epochs
    lr = args.lr
    betas = args.betas
    eps = args.eps
    weight_decay = args.weight_decay

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = Customized_CIFAR10_Dataset(root=data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0")
    model = iformer_small(pretrained=False)
    if torch.cuda.device_count() > 1:
        print("Data parallel in ", torch.cuda.device_count(), "GPUs")
        model = DP(model)
    else:
        print("Running on 1 GPU")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()

        correct = 0
        total = 0
        data_loading_time = 0.0
        training_time = 0.0

        start = data_loading_checkpoint = perf_counter()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            data_loading_time += (perf_counter() - data_loading_checkpoint)

            training_start_time = perf_counter()

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            training_time += (perf_counter() - training_start_time)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            data_loading_checkpoint = perf_counter()

        running_time = perf_counter() - start
        accuracy = 100. * correct / total

        print(f"Epoch {epoch}/{epochs} ")
        print(f"Data loading time (sec) is {data_loading_time:.3f}")
        print(f"Training time (sec) is {training_time:.3f}")
        print(f"Running time (sec) is {running_time:.3f}")
        print(f"Accuracy is {accuracy}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument("--num_workers", type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='/scratch/tn2151/project/cifar10-128',
                        help='Path to the training data')
    parser.add_argument("--epochs", type=int, default=5,
                        help='Use the first 4 epochs as warmup')
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999),
                        help="Betas for AdamW")
    parser.add_argument("--eps", type=float, default=1e-08,
                        help="Epsilon for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Need at least 1 GPU"

    train(args)
