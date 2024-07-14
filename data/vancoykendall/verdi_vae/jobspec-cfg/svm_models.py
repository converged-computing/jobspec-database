#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn import svm
from PIL import Image
from torchvision import transforms
to_tensor = transforms.ToTensor()
import os
import glob
from tqdm import tqdm
from disvae.utils.modelIO import load_model
import pandas as pd
import pickle
import sys
from sklearn.model_selection import GridSearchCV

# Grab the arguments that are passed in
t_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

M_NAMES = ['z'+str(i) for i in range(10,21,2)]
M_NAMES = [M_NAMES[t_id-1]] 

# In[255]:


# Load in the torch vae models
# model_name = "btcvae_cardamage_128_z10"
# model_dir = os.path.join("results", model_name)
# epoch = 0
# model_epoch = "model-{}".format(epoch) #
# vae_model = load_model(model_dir) #, filename=model_epoch)
m_paths = [os.path.join("results", "btcvae_cardamage_128_"+name) for name in M_NAMES]
model_dict = {}
print("loading models...", M_NAMES)
for m_path in m_paths:
    m_name = m_path.split('_')[-1]
    model_dict[m_name] = load_model(m_path)


# In[53]:


data_root = "data/car_damage_128"
img_paths = []
for ext in ['.jpg', ".JPEG"]:
    img_paths.extend(glob.glob(os.path.join(data_root, "*/*" + ext)))


# In[238]:


def create_data(vae_model, img_paths=img_paths):
    Z = list()
    y = np.empty(shape=len(img_paths), dtype=int)
    for i, img_path in tqdm(enumerate(img_paths)):
        img = to_tensor(Image.open(img_path))
        img = img.unsqueeze(0)
        z_mean, z_log_var = vae_model.encoder.forward(img)
        z_mean = np.squeeze(z_mean.detach().numpy())
        Z.append(z_mean)
        label = img_path.split('/')[2] # expected path structure: data/<dataset>/<label>/<file>
        label = 0 if label == "Real" else 1
        y[i] = label
    Z = np.array(Z)
    return Z, y


# In[251]:


# Split into train/valid/test datasets
def data_split(Z, y, train_ratio=.75):
    """Splits data into train/valid/test arrays"""
    valid_ratio = (1-train_ratio)/2
    data = np.concatenate((Z, np.expand_dims(y, axis=1)), axis=1)
    np.random.shuffle(data)
#     train, valid, test = np.split(data, [int(train_ratio*data.shape[0]), int(valid_ratio*data.shape[0])])
    train, test = np.split(data, [int(train_ratio*data.shape[0])])
    
    X_train, y_train = np.split(train, [-1], axis=1)
#     X_valid, y_valid = np.split(valid, [-1], axis=1)
    X_test, y_test = np.split(test, [-1], axis=1)
    return X_train, np.squeeze(y_train), X_test, np.squeeze(y_test)
#     return X_train, np.squeeze(y_train), X_valid, np.squeeze(y_valid), X_test, np.squeeze(y_test)


# In[259]:


# Create data dicts for all models
train_dict = dict()
# valid_dict = dict()
test_dict = dict()
print("creating data...", M_NAMES)
for m_name in M_NAMES:
    X, y = create_data(model_dict[m_name])
#     X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(X, y, .6)
    X_train, y_train, X_test, y_test = data_split(X, y, .6)
    train_dict[m_name] = (X_train, y_train)
#     valid_dict[m_name] = (X_valid, y_valid)
    test_dict[m_name] = (X_test, y_test)
    with open("results/btcvae_cardamage_128_{}/svm_train.pkl".format(m_name), "wb") as file:
        pickle.dump((X_train, y_train), file)
    with open("results/btcvae_cardamage_128_{}/svm_test.pkl".format(m_name), "wb") as file:
        pickle.dump((X_test, y_test), file)


# In[261]:


# Train an SVM to classify the latent space between real and fake images from car damage dataset
Cs = [2**x for x in range(-5,15)]
gammas = [2**x for x in range(-15,3)]
# svm_dict = dict()
for m_name in M_NAMES:
    m = GridSearchCV(svm.SVC(), param_grid={'C': Cs, "gamma": gammas})
    X_train, y_train = train_dict[m_name]
    print("training...", m_name)
    m.fit(X_train, y_train)
    with open("results/btcvae_cardamage_128_{}/svm.pkl".format(m_name), "wb") as file:
        pickle.dump(m, file)
#     svm_dict[m_name] = m
    print(m_name, "fit")




