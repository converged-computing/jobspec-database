import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, random_split, DataLoader
import os
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from ResNet import resmodel, Class_DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.is_available() # Use gpu

"""
Number of correct predictions / total samples
"""

def accuracy(model, data):
    train_loader = torch.utils.data.DataLoader(data)
    if torch.cuda.is_available():
        model.cuda()

    i = 0
    correct = 0
    for imgs, labels in iter(train_loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        prediction = torch.sigmoid(model.forward(imgs.float()))
        prediction = (prediction > 0.5).float()
        groundtruths = labels 
        pred = np.squeeze(prediction.cpu().detach().numpy()).argmax()
        groundtruth =  np.squeeze(groundtruths.cpu().detach().numpy()).argmax()
        correct += int(pred == groundtruth)
        i += 1
    
    return correct/i

if __name__ == "__main__":
    r = A.Compose(
        [
            A.Resize(height=90, width=90),
            ToTensorV2(),
        ],
    )
    
    training = Class_DataLoader("/home/csc490w/csc490project/data/classification_dataset/training_images", "/home/csc490w/csc490project/data/classification_dataset/training_labels_classification.csv", r)
    model = resmodel()
    model.load_state_dict(torch.load("/home/csc490w/csc490project/code/saved_models/classResNet_modelBATCH100It100.pth"))

    f = open("/home/csc490w/csc490project/code/saved_models/accuracies_resnet.txt", "w")
    acc = accuracy(model, training) 
    f.write("training " + str(acc) + "\n")
    f.close()