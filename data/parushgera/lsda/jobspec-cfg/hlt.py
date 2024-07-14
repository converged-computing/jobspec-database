#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import re
import os
import nltk
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
SEED = 1013
np.random.seed(SEED)
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples 
from stance_utils import *
from stance_models import * 
#from parameters import *
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dropout,Concatenate,Dense, Embedding, SpatialDropout1D, Flatten, GRU, Bidirectional, Conv1D,MaxPooling1D

from tensorflow.keras.layers import RNN, Dropout,Concatenate,Dense, Embedding,LSTMCell, LSTM, SpatialDropout1D, Flatten, GRU, Bidirectional, Conv1D, Input,MaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from sklearn.model_selection import StratifiedKFold
stemmer = PorterStemmer()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stopwords_english = stopwords.words('english')
from sklearn.preprocessing import LabelEncoder
import keras.backend as K
from keras.layers import Lambda
import random
import matplotlib.pyplot as plt


# In[2]:


file =open('/data/parush/wtwt/wtwt_extracted.json','r')
data = json.load(file)
df = pd.DataFrame(data)
df = df.drop(columns = 'tweet_id')
df = df.rename(columns={"merger": "target"})
df['domain'] = np.where(df['target']== 'FOXA_DIS', 'entertainment', 'healthcare')
df


# In[ ]:


mode = 'in'


# In[3]:


classes = {'support': np.array([1, 0, 0, 0]), 'refute': np.array([0, 1, 0, 0]), 'comment': np.array([0, 0, 1, 0]), 'unrelated': np.array([0, 0, 0, 1])}
classes_ = np.array(['support', 'refute', 'comment', 'unrelated'])


# In[4]:


def in_splitter(df, domain):
    df = df[df['domain'] == domain]
    X = df[['target','tweet', 'domain']]
    y = df[['stance']]
    sentence_maxlen = 0
    target_maxlen = 0
    x_s_token = []
    x_s_test_token = []
    x_t_token = []
    x_t_test_token = []
    y_train = []
    y_test = []
    
    X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True,stratify = y)
    print("Started splitting on {}, pre-processing".format(domain))
    for tweet in X_train['tweet'].values:
        tweet = process_tweet(tweet)
        if len(tweet) > sentence_maxlen:
            sentence_maxlen = len(tweet)
        x_s_token.append(tweet)
        
    for target in X_train['target'].values:
        if len(target) > target_maxlen:
            target_maxlen = len(target)
        x_t_token.append(target)
    for target in X_test['target'].values:
        if len(target) > target_maxlen:
            target_maxlen = len(target)
        x_t_test_token.append(target)
    
    for tweet in X_test['tweet'].values:
        tweet = process_tweet(tweet)
        if len(tweet) > sentence_maxlen:
            sentence_maxlen = len(tweet)
        x_s_test_token.append(tweet)

    for i in y_train_.values:
        y_train.append(classes[i[0]]) # fix this
    for i in y_test_.values:
        y_test.append(classes[i[0]])
    
    return x_s_token, x_t_token, y_train, x_s_test_token, x_t_test_token,  y_test, sentence_maxlen, target_maxlen
    
    


# In[ ]:





# In[5]:


def cross_splitter(df, source, destination):
    df_source = df[df['domain'] == source]
    df_destination = df[df['domain'] == destination]
    
    sentence_maxlen = 0
    target_maxlen = 0
    x_s_token = []
    x_s_test_token = []
    x_t_token = []
    x_t_test_token = []
    y_train = []
    y_test = []
    
    #X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True,stratify = y)
    print('Training on {} and testing on {}'.format(source,destination))
    print("Started splitting, pre-processing")
    for tweet in df_source['tweet'].values:
        tweet = process_tweet(tweet)
        if len(tweet) > sentence_maxlen:
            sentence_maxlen = len(tweet)
        x_s_token.append(tweet)
        
    for target in df_source['target'].values:
        if len(target) > target_maxlen:
            target_maxlen = len(target)
        x_t_token.append(target)
    for target in df_destination['target'].values:
        if len(target) > target_maxlen:
            target_maxlen = len(target)
        x_t_test_token.append(target)
    
    for tweet in df_destination['tweet'].values:
        tweet = process_tweet(tweet)
        if len(tweet) > sentence_maxlen:
            sentence_maxlen = len(tweet)
        x_s_test_token.append(tweet)

    for i in df_source['stance'].values:
        y_train.append(classes[i]) # fix this
    for i in df_destination['stance'].values:
        y_test.append(classes[i])
    
    return x_s_token, x_t_token, y_train, x_s_test_token, x_t_test_token,  y_test, sentence_maxlen, target_maxlen
    
    


# In[6]:


domains = {'ent': 'entertainment', 'hlt':'healthcare'}

if mode == 'in':
    x_s_token, x_t_token, y_train, x_s_test_token, x_t_test_token,  y_test, sentence_maxlen,target_maxlen= in_splitter(df, domains['hlt'])
if mode == 'cross':
    x_s_token, x_t_token, y_train, x_s_test_token, x_t_test_token,y_test, sentence_maxlen,target_maxlen= cross_splitter(df, domains['hlt'], domains['ent'] )


# In[ ]:





# In[7]:


vocabulary = build_vocab(x_s_token + x_t_token )
vocab_size = len(vocabulary)
print("Total words in vocab are",vocab_size)
embedding_matrix = get_embeddings('twitter',100,vocabulary)
print('X_S_token {}, y_train {}, y_test {}'.format(len(x_s_token), len(y_train), len(y_test)))


# In[8]:


batch_size = 32
epochs = 50
units = 60
opt = keras.optimizers.Adam(learning_rate=1e-3)
num_classes = 4
bicondORnot = False # Set True in case of using bicond model


# In[ ]:





# In[9]:


if bicondORnot:
    y_train = np.asarray(y_train)
    _,balance = divmod(len(y_train),batch_size)
    balance = batch_size-balance
    y_train = list(y_train)
    for i in range(balance):
        index = np.random.randint(1, len(y_train))
        x_s_token.append(x_s_token[index])
        y_train.append(y_train[index])
        x_t_token.append(x_t_token[index])


# In[10]:


x_s = [tweet_to_tensor(each_s,vocabulary) for each_s in x_s_token]
x_s = pad_sequences(x_s, maxlen = sentence_maxlen, padding = 'post')
x_s_test = [tweet_to_tensor(each_s,vocabulary) for each_s in x_s_test_token]
x_s_test = pad_sequences(x_s_test, maxlen = sentence_maxlen, padding = 'post')

x_t = [tweet_to_tensor(each_s,vocabulary) for each_s in x_t_token]
x_t = pad_sequences(x_t, maxlen = sentence_maxlen, padding = 'post')
x_t_test = [tweet_to_tensor(each_s,vocabulary) for each_s in x_t_test_token]
x_t_test = pad_sequences(x_t_test, maxlen = sentence_maxlen, padding = 'post')
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_s = x_s[shuffle_indices]
x_t = x_t[shuffle_indices]
y_train = np.asarray(y_train)
y_train = y_train[shuffle_indices]
y_test = np.asarray(y_test)


# In[11]:


# Select the model from below......


model = biLSTMCNN(embedding_matrix, num_classes, sentence_maxlen)
#model = biLSTM(embedding_matrix, num_classes)
#model = bicond(units, opt, embedding_matrix, x_t, batch_size, sentence_maxlen,num_classes)


# In[12]:


model.summary()


# In[13]:


if model.name == 'bicond': # making test_set % batch_size = 0 and validation_split % batch_size = 0
    print(len(y_test))
    v_num = len(y_train)//10 
    print('There are {} train examples'.format(len(y_train)))
    _, b3 = divmod(v_num,batch_size)
    v_split = (v_num  + (batch_size-b3)) / len(y_train)
    history = model.fit(x_s, y_train, epochs = epochs, batch_size = batch_size,validation_split = v_split,  verbose=1)
    l = len(x_s_test)
    _, balance2 = divmod(l,batch_size)
    print()
    x_s_test = list(x_s_test)
    fill_number = batch_size - balance2
    print('Total number of test examples are {}, and fill number is {}'.format(l, fill_number))
    for i in range(fill_number):
        x_s_test.append(np.zeros(sentence_maxlen,))

    x_s_test = np.array(x_s_test)
    print('length of x_s_test ', len(x_s_test))
    y_pred = np.round(model.predict(x_s_test, batch_size = batch_size))
    print('length of y_pred ', len(y_pred))
    print(classification_report(y_test, y_pred[:-fill_number], digits=4, labels = [0,1]))
else:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cvscores = []

    for train, val in kfold.split(x_s, classes_[y_train.argmax(1)]):
        history = model.fit(x_s[train], y_train[train], epochs = epochs, batch_size = batch_size, verbose=1)
        scores = model.evaluate(x_s[val], y_train[val], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    y_pred = np.round(model.predict(x_s_test))
    
    print(classification_report(y_test, y_pred, digits=4))


