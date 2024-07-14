print("on line 1")
import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

print('modules loaded')

data = np.load('/blue/eee4773/eric.fowler/Final-Project/TeamPip-PyTorch/Data/data_train.npy').T
X_og = data.copy()/255
y =np.loadtxt('/blue/eee4773/eric.fowler/Final-Project/TeamPip-PyTorch/Data/correct_labels.npy')
y = y.astype(float).astype(int)

print(np.shape(data))
print(np.shape(y))

#X = X_og.copy()
def resize_func(input_data,new_width,new_height):#input the (1,90000) data
    size1 = np.shape(input_data)[0]
    size2 = int(size1**(0.5))
    output = cv2.resize(input_data.reshape(size2,size2),(new_width,new_height))
    return output
def morph_ops(input_data):
    dilate_kernel = np.ones((2,2),np.uint8)
    open_kernel = np.ones((3,3),np.uint8)
    closed_kernel = np.ones((3,3),np.uint8)
    d2_kernel = np.ones((3,3),np.uint8)
    gradient_kernel = np.ones((3,3),np.uint8)
    open2_kernel = np.ones((3,3),np.uint8)
    picture = resize_func(input_data,250,250)
    dilated_picture = cv2.dilate(picture,dilate_kernel,iterations=4)
    opened_picture = cv2.morphologyEx(dilated_picture,cv2.MORPH_OPEN,open_kernel)
    closed_picture = cv2.morphologyEx(opened_picture,cv2.MORPH_CLOSE,closed_kernel)
    d2_picture = cv2.dilate(closed_picture,d2_kernel,iterations=1)
    gradient_picture = cv2.morphologyEx(d2_picture,cv2.MORPH_GRADIENT,gradient_kernel)
    open2_picture = cv2.morphologyEx(gradient_picture,cv2.MORPH_OPEN,open2_kernel)
    return open2_picture

X_new = np.zeros((np.shape(X_og)[0],300*300))
print(np.shape(X_og))
print(np.shape(X_new))
for ii in range(np.shape(X_new)[0]):
    newrow = morph_ops(X_og[ii,:])
    newrow = newrow.reshape(1,-1)
    newrow = resize_func(newrow[0,:],300,300)
    newrow = newrow.reshape(1,-1)
    X_new[ii,:] = newrow
print(np.shape(X_new))
#X = X_new
X =  1-X_og



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


BATCH_SIZE = 200

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

print('Before MLP')
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(90000,250)
        self.linear2 = nn.Linear(250,100)
        self.linear3 = nn.Linear(100,10)
    
    def forward(self,X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)
 
mlp = MLP()
print(mlp)

def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5000
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
                
fit(mlp, train_loader)

def evaluate(model):
#model = mlp
    correct = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct)*100 / (len(test_loader)*BATCH_SIZE)))
evaluate(mlp)


torch_X_train = torch_X_train.view(-1, 1,300,300).float()
torch_X_test = torch_X_test.view(-1,1,300,300).float()
print(torch_X_train.shape)
print(torch_X_test.shape)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3)
        self.fc1 = nn.Linear(71*71*128,250)
        self.fc2 =nn.Linear(250,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,128*71*71)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
cnn = CNN()
print(cnn)

it = iter(train_loader)
X_batch, y_batch = next(it)
print(X_batch.shape)
print(cnn.forward(X_batch).shape)

fit(cnn,train_loader)
evaluate(cnn)
