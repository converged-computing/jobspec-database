import sys
sys.path.append('framework')
from NetworkClass import Network
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
import time
import argparse
from Pruning import Pruning

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pruning_perc', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--prune_iter', type=int, default=2)



args = parser.parse_args()



class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)



# Define Constants
batch_size_train = 64
batch_size_test = 1000
n_epochs = args.epochs
log_interval = 10
random_seed = 1




# Model Definition
model_dict = {
        "network":{
            'input_layer': {
                "units": 784,
                
                },
            'hidden_layer': [{
                    "units": 500, 
                    "activation": "relu",
                    "type": "Linear"
                }, 
                {
                    "units": 300, 
                    "activation": "relu",
                    "type": "Linear"

                }],
            'output_layer': {
                "units": 10,
                "activation": "softmax",
                "type": "Linear"
                }
        }
    }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network(model_dict)
model = nn.DataParallel(model).to(device)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)), ReshapeTransform((-1,))
                             ])),
  batch_size=batch_size_train, shuffle=True)


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)), ReshapeTransform((-1,))
                             ])),
  batch_size=batch_size_test, shuffle=True)



# Define Loss and Optimizer



def train(mod, optim, epoch):
  mod.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optim.zero_grad()
    output = mod(data)
    loss = criterion(output, target)
    loss.backward()
    optim.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))



def test(mod):
  mod.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = mod(data)
      test_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  return test_loss, 100. * correct / len(test_loader.dataset)




paramList = []

for lr in np.arange(0.01, 0.1, 0.01):
  for prune_perc in np.arange(0.1, 0.8, 0.05):
    for prune_iter in np.arange(2, int(n_epochs/2)):
      print("Start Grid Search with Params: LR: {} \tPrune Percentage: {} \t Prune Iterations: {}".format(lr, prune_perc, prune_iter))
      learning_rate = 0.01
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=lr)

      pruning = Pruning(percentage=prune_perc)

      total_time = 0
      losses = []
      accuracies = []
      times = []
      for epoch in range(1, n_epochs + 1):
        start_t = time.time()
        train(model, optimizer, epoch)
        total_time += time.time() - start_t
        loss, acc = test(model)  
     
        losses.append(loss)
        accuracies.append(acc)
        times.append(total_time)

        if epoch % prune_iter == 0:
          
          optimizer, model = pruning.prune_model(optimizer, model)
      
      grid_dict = {}
      grid_dict["loss"] = losses
      grid_dict["accuracy"] = accuracies
      grid_dict["train_time"] = times
      grid_dict["lr"] = lr
      grid_dict["prune_iter"] = prune_iter
      grid_dict["prune_perc"] = prune_perc


      paramList.append(grid_dict)

paramDf = pd.DataFrame(paramList)
paramDf.to_csv("{}.csv".format(time.time()))
