"""This script trains a NeuralNetClassifier on CIFAR-10 data,
using skorch and dask for multi-GPU hyperparameter optimization.
"""

import argparse
import time
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from skorch import NeuralNetClassifier

from pt_resnet_cifar10.resnet import ResNet
from pt_resnet_cifar10.resnet import test

from dask.distributed import Client
from joblib import parallel_backend

LOCAL_STORAGE = os.getenv('$SCRATCH')
SAVE_PATH = "/home/c7wilson/scratch/cifar_tests/"

def get_data():

    path = '/home/c7wilson/scratch/datasets/cifar10'
    print(f'Loading data from {path}...')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_ds = datasets.CIFAR10(root=path, train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,]),
        download=False)

    X_train = np.moveaxis(train_ds.data.astype('float32'), -1, 1)
    y_train = np.array(train_ds.targets)

    test_ds = datasets.CIFAR10(root=path, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,]),
        download=False)

    X_test = np.moveaxis(test_ds.data.astype('float32'), -1, 1)
    y_test = np.array(test_ds.targets)

    return X_train, X_test, y_train, y_test

def main(device, batch_size, lr, max_epochs):

    client = Client('127.0.0.1:8786')

    X_train, X_test, y_train, y_test = get_data()
    # trigger potential cuda call overhead
    torch.zeros(1).to(device)

    print("\nTesting skorch performance")
    tic = time.time()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    net = NeuralNetClassifier(
        ResNet,
        batch_size=batch_size,
        optimizer=torch.optim.Adadelta,
        criterion=torch.nn.CrossEntropyLoss,
        lr=lr,
        device=device,
        max_epochs=max_epochs
    )

    params = {
        'module__num_blocks': [[3,3,3], [5,5,5], [7,7,7]]
    }

    gs = GridSearchCV(net, params, scoring='accuracy', cv=5, verbose=3, refit=True)
    
    with parallel_backend('dask'):
        gs.fit(X_train, y_train)
    print(gs.cv_results_)

    y_pred = gs.best_estimator_.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    time_skorch = time.time() - tic

    print(f'Grid search found model with validation score: {gs.best_score_}')
    print(f'with parameters: {gs.best_params_}')
    print(f'Test score: {score} after {max_epochs} in {time_skorch}s.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="skorch CIFAR10 benchmark")
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (e.g. "cuda", "cpu")')
    parser.add_argument('--max_epochs', type=int, default=12,
                        help='max epochs to run for')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=int, default=0.1,
                    help='Batch size for training')
    args = parser.parse_args()
    main(device=args.device, batch_size=args.batch_size, lr=args.lr, max_epochs=args.max_epochs)
