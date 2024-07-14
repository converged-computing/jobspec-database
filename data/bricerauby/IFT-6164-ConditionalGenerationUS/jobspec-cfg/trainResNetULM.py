from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
from trainer.functionnalTrainingClassifier import train, test, adjust_learning_rate
import glob
import numpy as np
from models import *
from dataset.ClassifierDataset import ClassifierDataset
from display.functionnalDisplay import display_random_samples

dataPrefix = os.path.join(os.environ.get('SLURM_TMPDIR'),'patchesIQ_small_shuffled')
learning_rate = 0.1
momentum = 0.9
weight_decay=0.0002
batch_size_train= 512
batch_size_eval=400
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

experiment = Experiment(project_name='cgenulm',
                            workspace='bricerauby', auto_log_co2=False)
experiment.set_name(os.environ.get('SLURM_JOB_ID') + '_' + experiment.get_name())
experiment.add_tag('resnet-18')
code_list = glob.glob("**/*.py", recursive=True)

train_dataset = ClassifierDataset(dataPrefix, 'trainMB.h5', 'trainNoMB.h5',num_frames=16)
test_dataset = ClassifierDataset(dataPrefix, 'testMB.h5', 'testNoMB.h5',num_frames=16)

display_random_samples(train_dataset, num_samples=5, experiment=experiment)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=6,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=6,pin_memory=True)

net = ResNet18(in_chans=1)
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)



for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch, learning_rate, decay=[10,15])
    train(train_loader, net, epoch, experiment, optimizer,criterion, device)
    test(test_loader, net, epoch, experiment,criterion, device)