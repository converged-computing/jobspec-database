# -*- coding: utf-8 -*-
"""
CS834_Fall_2023_Disaster_Tweets_Project__
|
lemos_kerasnlp_for_slurm_job_training.py
Created on Wed Nov  8 17:45:29 2023
@author: Rochana Obadage
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_core as keras
import keras_nlp

import pytz
from datetime import datetime
import re

import argparse
import sys

from nltk.tokenize import word_tokenize,sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer

from contractions import fix 
from unidecode import unidecode

# remove newlines , spaces
def remove_blank(data):
    text=data.replace("\\n"," ").replace("\t"," ")
    return text

# Contractions mapping
def contract_text(data):
    text=fix(data)
    return text

# handling accented character
def handling_accented_chr(data):
    text=unidecode(data)
    return text

# remove stopwords
stopwords_list=stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('nor')
stopwords_list.remove('not')

# clean the text
def clean_text(data):
    # tokenization
    tokens=word_tokenize(data)
    # lowercase the data
    #lowercase=[word.lower() for word in tokens]
    # remove stopwords
    #remove_stopwords=[word for word in lowercase if word not in stopwords_list]
    # remove punctuations
    #remove_punctuations=[word for word in remove_stopwords if word not in punctuation]
    # remove len(word)<2
    #remove_len_word=[word for word in remove_punctuations if len(word)>2]
    # word contains only alphabet not number
    #final_text=[word for word in remove_len_word if word.isalpha()]
    
    clean_data=[i.lower() for i in tokens if (i.lower() not in punctuation) and (i.lower() not in stopwords_list) and (len(i)>2) and (i.isalpha())]

    
    return clean_data

def lemmatization(data):
    final_text=[]
    lemmatizing=WordNetLemmatizer()
    for i in data:
        lemma=lemmatizing.lemmatize(i)
        final_text.append(lemma)
        
    return " ".join(final_text)

#-----------------------------------------------

def pre_process(text):
    text = re.sub(r"\n","",text)
    text = text.lower()
    text = re.sub(r"\d","",text)        #Remove digits
    text = re.sub(r'[^\x00-\x7f]',r' ',text) # remove non-ascii
    text = re.sub(r'[^\w\s]','',text) #Remove punctuation
    text = re.sub(r'http\S+|www.\S+', '', text) #Remove http
    return text


def run_training(itr,epoch=2):


    ITERATION = itr
    epochs = int(epoch)
    print(epochs)
    
    dataset_name = 'lemos_DT_nlp'
    saved_model_path = './saved_models/{}_bert_{}.keras'.format(dataset_name.replace('/', '_'),ITERATION)
    
    
    print("TensorFlow version:", tf.__version__)
    print("keras_core version:", keras.__version__)
    print("KerasNLP version:", keras_nlp.__version__)
    print("NumPy version:", np.__version__)
    print("pandas version:", pd.__version__)
    
    df_train = pd.read_csv("dataset/extended_training_dataset.csv")  #changed from "train.csv"
    df_test = pd.read_csv("dataset/test.csv")
    df_local_test = pd.read_csv("dataset/validation_split.csv")
    
    print('Training Set Shape = {}'.format(df_train.shape))
    print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
    print('Test Set Shape = {}'.format(df_test.shape))
    print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))
    
    BATCH_SIZE = 32
    NUM_TRAINING_EXAMPLES = df_train.shape[0]
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    STEPS_PER_EPOCH = int(NUM_TRAINING_EXAMPLES)*TRAIN_SPLIT // BATCH_SIZE
    
    EPOCHS = epochs #2
    AUTO = tf.data.experimental.AUTOTUNE

    # pre-processing method 1
    # df_train['text'] = df_train['text'].apply(lambda x : pre_process(x))    

    # pre-processing NLTK
    df_train['text']=df_train['text'].apply(remove_blank)
    df_train['text']=df_train['text'].apply(contract_text)
    df_train['text']=df_train['text'].apply(handling_accented_chr)
    df_train['text']=df_train['text'].apply(clean_text)
    df_train['text']=df_train['text'].apply(lemmatization)

    # pre-processing NLTK for local testing
    df_local_test['text']=df_local_test['text'].apply(remove_blank)
    df_local_test['text']=df_local_test['text'].apply(contract_text)
    df_local_test['text']=df_local_test['text'].apply(handling_accented_chr)
    df_local_test['text']=df_local_test['text'].apply(clean_text)
    df_local_test['text']=df_local_test['text'].apply(lemmatization)

    print("\ndf_train text pre processed\n")
    
    X = df_train["text"]
    y = df_train["target"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)
    
    X_test = df_test["text"]

    #     # Pretrained classifier.
    # classifier = keras_nlp.models.BertClassifier.from_preset(
    #     "bert_base_en_uncased",
    #     num_classes=4,
    # )
    # classifier.fit(x=features, y=labels, batch_size=2)
    # classifier.predict(x=features, batch_size=2)
    
    # # Re-compile (e.g., with a new learning rate).
    # classifier.compile(
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     optimizer=keras.optimizers.Adam(5e-5),
    #     jit_compile=True,
    # )

    
    # Load a DistilBERT model
    preset= "distil_bert_base_en_uncased"
    
    # Use a shorter sequence length
    preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(preset,
                                                                       sequence_length=160,
                                                                       name="preprocessor_4_tweets"
                                                                      )
    
    # Pretrained classifier
    classifier = keras_nlp.models.DistilBertClassifier.from_preset(preset,
                                                                   preprocessor = preprocessor,
                                                                   num_classes=2)
    
    classifier.summary()
    
    # Compile
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), #'binary_crossentropy',
        optimizer=keras.optimizers.Adam(1e-5),
        metrics= ["accuracy"]
    )
    
    # Fit
    history = classifier.fit(x=X_train,
                             y=y_train,
                             batch_size=BATCH_SIZE,
                             epochs=EPOCHS,
                             validation_data=(X_val, y_val)
                            )
    
    
    

    # X = df_train["text"]
    # y = df_train["target"]
    
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)
    
    # X_test = df_test["text"]
    classifier.save(saved_model_path)


   
    # # calcuating the F1 for new model with validation dataset (local testing)
    X_test = df_local_test["text"]
    y_test_true = df_local_test['target']

    y_pred = classifier.predict(X_test) #X_train == validation_set
    # displayConfusionMatrix(y_train, y_pred_train, "Training")

    tn, fp, fn, tp = confusion_matrix(y_test_true, np.argmax(y_pred, axis=1)).ravel()
    f1_score = tp / (tp+((fn+fp)/2))

    tz_VA = pytz.timezone('America/Virgin')
    datetime_VA = datetime.now(tz_VA)

    # only write when f1_curr > f1_prev
    

    # if f1_score > f1_prev :
    #     classifier.save(saved_model_path)
        
    with open('results/performances/model_performances.txt','a') as f:
        f.write('\n')
        f.write(str(datetime_VA.strftime("%y_%m_%d_%H_%M_%S")))
        f.write('\n')
        f.write(saved_model_path)
        f.write('\n')
        records_used_for_training = f'records_used_for_training:{len(df_train)}'
        f.write(records_used_for_training)
        f.write('\n')
        f.write(str(f1_score))
        f.write("\n\n")


    


    # disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))    
    # f1 = 0.7
    
    print("Successfully saved at :\n",saved_model_path)




def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-itr", "--iteration", help="training iteration")
    argParser.add_argument("-ep", "--epochs", help="training epochs")
    args = argParser.parse_args()

    itr = args.iteration
    epochs = args.epochs
    
    run_training(itr,epochs)
    

if __name__ == "__main__":
    main()

