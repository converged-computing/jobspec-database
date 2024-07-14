# CNN to classifiy genomic sequences as closed or open based on ATAC seq data
# Daniel Lyon | WUSTL Cohen Lab | Jan, 2024

import os
import argparse
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tf_tools import cnn_classifier, cnn_regression



def one_hot_seqs(seqs) -> np.array:
    static_1hotmap = {
        'A' : np.array([1,0,0,0]),
        'a' : np.array([1,0,0,0]),
        'C' : np.array([0,1,0,0]),
        'c' : np.array([0,1,0,0]),
        'G' : np.array([0,0,1,0]),
        'g' : np.array([0,0,1,0]),
        'T' : np.array([0,0,0,1]),
        't' : np.array([0,0,0,1]),
    }
    onehot_seqs = []
    for seq in seqs:
        onehot_seqs.append(
            [static_1hotmap[seq[i]] if seq[i] in static_1hotmap.keys() else static_1hotmap[random.choice(['A','C','G','T'])] for i in range(len(seq))]
        )
    return np.stack(onehot_seqs)



def main(output_dir, data_file, num_classes, batch_size, epochs, fold, FEATURE_KEY, LABEL_KEY, lr):
    
    keras.utils.set_random_seed(13*fold+7*fold+1)
    
    #Read and split up data into train, validate, test based on fold
    filename, file_extension = os.path.splitext(data_file)
    if file_extension == 'parquet':
        data_df = pd.read_parquet(data_file)
    else:
        data_df = pd.read_csv(data_file, index_col=0)
        
    seq_len = len(data_df.iloc[0][FEATURE_KEY])
    
    test_df = data_df[data_df['test_set']].sample(frac=1)
    validation_df = data_df[data_df['validation_set']]
    train_df = data_df[data_df['train_set']]
    
    print(len(train_df), ": training points", flush=True)
    print(len(validation_df), ": validation points", flush=True)
    print(len(test_df), ": reserved testing points", flush=True)
    
    #############################################################
    # Prepare data for fitting
    x_train = one_hot_seqs(train_df[FEATURE_KEY])
    x_validation = one_hot_seqs(validation_df[FEATURE_KEY])
    x_test = one_hot_seqs(test_df[FEATURE_KEY])
    
    # encoder = LabelEncoder()
    # encoder.fit(data_df[LABEL_KEY])
    # classes = encoder.classes_
    # num_classes = len(classes)
    # y_train = encoder.transform(train_df[LABEL_KEY])
    # y_validation = encoder.transform(validation_df[LABEL_KEY])
    # y_test = encoder.transform(test_df[LABEL_KEY])

    
    
    y_train = train_df[LABEL_KEY].values
    y_validation = validation_df[LABEL_KEY].values
    y_test = test_df[LABEL_KEY].values
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_validation = keras.utils.to_categorical(y_validation, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    ##############################################################
    # Create model and callbacks then fit
    
    # Tensorboard setup
    tensor_logs = os.path.join(output_dir, "tb_logs")
    os.makedirs(tensor_logs, exist_ok=True)
    tensorboard_cb = keras.callbacks.TensorBoard(tensor_logs, histogram_freq=1)
    
    # Early stopping setup
    earlystop_cb = keras.callbacks.EarlyStopping('val_loss', patience=15)
    
    livestream_cb = keras.callbacks.CSVLogger(filename = os.path.join(output_dir, "live_log.csv"))

    # Load model
    seq_len = len(data_df.iloc[0][FEATURE_KEY])
    model = cnn_regression.bestResNet(input_shape=(seq_len,4),lr=lr,bins=num_classes,output='softmax',loss='categorical_crossentropy')
    
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        batch_size=batch_size,
        callbacks =[earlystop_cb, tensorboard_cb, livestream_cb],
        verbose=0,
    )
    
    metrics = pd.DataFrame(
        history.history,
        index = np.arange(len(history.epoch))+1
    )
    
    metrics.to_csv(
        os.path.join(output_dir,'training_history.csv'),
        index_label='epoch'
    )
    
    
    ###########################################################
    # Test model performance and get score metrics
    
    model.save(os.path.join(output_dir, "cnn_model.keras"))
    
    # y_pred1 = model.predict(x_test, verbose=0)
    # y_pred = np.argmax(y_pred1, axis=1)
    # print("Test Set Report")
    # print(classification_report(y_test, y_pred,target_names=['open','closed']), flush=True)
    
    return
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", type=str, help='Where to write the stuff')
    parser.add_argument("data_file", type=str, help='Path to file with features and lables')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--FEATURE_KEY", type=str, default='sequence', help="Column name(s) fo feature for model input")
    parser.add_argument("--LABEL_KEY", type=str, default='open')
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args = parser.parse_args()
    
    main(**vars(args))










