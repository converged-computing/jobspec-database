# Trains iterative CNN Regression Models to evaluate performance of node dropout uncertainty testing
# Daniel Lyon | WUSTL Cohen Lab | Feb, 2024

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import numpy as np
import pandas as pd
import random
import time

from scipy.io import savemat
from scipy.stats import linregress, spearmanr

import tensorflow as tf
from tensorflow import keras
from tf_tools import cnn_regression

nsamples = 100

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

def getNewData(model,df, feature_key,mode,nt,sz):
    #Ensure there is no overflow
    remaining_df = df[~df['used']]
    k = sz
    if k > len(remaining_df):
        k = len(remaining_df)
        if k==0:
            return []
        
    # If we are in Random mode
    if mode == 'Random':
        return remaining_df.sample(k).index.to_list()
    
    # Take nt predictions and compute their std
    predictions = []
    # batched_xt = batch_data(one_hot_seqs(df[feature_key]), 128)
    for _ in range(nt):
    #     p = []
    #     for batch in batched_xt:
    #         p.append(model(batch).numpy().flatten())
    #     predictions.append(np.concatenate(p))
        predictions.append(model.predict(one_hot_seqs(remaining_df[feature_key]), verbose=0).flatten())

    predictions = np.stack(predictions)
    preds_std = np.std(predictions, axis=0)
    
    # Take prediction std data and put it into a series.
    # Sort the series then get the index for the highest stds.
    # Remove those from the remaining data
    uncertainty_s = pd.Series(
        preds_std,
        index=remaining_df.index
    )
    
    print(f"Average Standard Deviation of remaining data:{np.mean(preds_std)}", flush=True)
    
    return uncertainty_s.sort_values(ascending=False).index[:k].to_list()

def saveTestMetrics(model, xt, yt, nt, out_dir):
    predictions = []
    # batched_xt = batch_data(xt, 128)
    for _ in range(nt):
    #     p = []
    #     for batch in batched_xt:
    #         p.append(model(batch).numpy().flatten())
    #     predictions.append(np.concatenate(p))
        predictions.append(model.predict(xt, verbose=0).flatten())
    predictions = np.stack(predictions)
    preds_mean = np.mean(predictions,axis=0)
    pcc = linregress(preds_mean, yt).rvalue
    scc = spearmanr(preds_mean,yt).statistic
    
    stats = pd.Series(
        [pcc,scc],
        index=['pcc','scc'],
        name='metrics'
    )
    stats.to_csv(
        os.path.join(out_dir, "test_metrics.csv")
    )
    print("Test Metrics:", flush=True)
    print(stats, flush=True)
    return
    
def batch_data(xt, batch_size):
    batched = []
    num_batches = int(len(xt/batch_size))
    i=0
    while i < num_batches:
        start = i*batch_size
        batched.append(xt[start:start+batch_size])
        i+=1
    start = i*batch_size
    batched.append(xt[start:])
    return batched



def main(output_dir, data_file, mode, iteration, batch_size, epochs, initial_data, sampling_size, num_test, fold, FEATURE_KEY, LABEL_KEY, lr):

    keras.utils.set_random_seed(13*fold+7*fold+1)
    
    iter_output_dir = os.path.join(output_dir,iteration)
    os.makedirs(iter_output_dir,exist_ok=True)
    
    
    # Read and split up data into train, validate, test based on fold
    data_df = pd.read_parquet(data_file)
    
    test_df = data_df[data_df['test_set']]
    validation_df = data_df[data_df['validation_set']]
    train_df = data_df[data_df['train_set']].sample(frac=1, random_state=fold+int(iteration))
    
    # Get the latest training data set
    if iteration == '0':
        training_i = train_df[train_df['data_batch_name'] == 'Genomic'].index
        if fold == 0:
            pass
            #TODO put writeout here
    else:
        iter_input_dir = os.path.join(output_dir, str(int(iteration)-1))
        attempts = 0
        while attempts < 20:
            try:
                with open(os.path.join(iter_input_dir, "next_index.line"), 'r') as file:
                    data = file.read()
            except:
                attempts += 1
                print("Next Index file not found on attempt", attempts, flush=True)
                time.sleep(1)
        training_i = data.strip().split(" ")
    
    train_df.loc[training_i, ['used']] = True
    
    # Prepare data for fitting
    x_validation = one_hot_seqs(validation_df[FEATURE_KEY])
    x_test = one_hot_seqs(test_df[FEATURE_KEY])
    
    y_validation = validation_df[LABEL_KEY].values
    y_test = test_df[LABEL_KEY].values
    
    
    ##############################################################
    # Create model and callbacks
    
    cbs = []
    
    # Tensorboard setup
    #tensor_logs = os.path.join(mode_output_dir, "tb_logs")
    #os.makedirs(tensor_logs, exist_ok=True)
    #tensorboard_cb = keras.callbacks.TensorBoard(tensor_logs, histogram_freq=1)
    #cbs.append(tensorboard_cb)
    
    # Early stopping setup
    earlystop_cb = keras.callbacks.EarlyStopping('val_loss', patience=15)
    cbs.append(earlystop_cb)
    
    # Livestream and subscribe
    #livestream_cb = keras.callbacks.CSVLogger(filename = os.path.join(mode_output_dir, "live_log.csv"))
    #cbs.append(livestream_cb)
        
    
    x_train = one_hot_seqs(train_df[train_df['used']][FEATURE_KEY])
    y_train = train_df[train_df['used']][LABEL_KEY].values
    
    print(f"------ Fold:{fold}, Mode:{mode}, Iteration:{iteration} ------", flush=True)
    print(f"Training with {len(train_df[train_df['used']])} data points", flush=True)
    print(f"Remaining training data: {len(train_df[~train_df['used']])}", flush=True)
    
    model = cnn_regression.originalResNet()
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        batch_size=batch_size,
        callbacks = cbs,
        verbose=0,
    )
    
    metrics = pd.DataFrame(
        history.history,
        index = np.arange(len(history.epoch))+1
    )
    print(metrics, flush=True)
    
    metrics.to_csv(
        os.path.join(iter_output_dir,'training_history.csv'),
        index_label='epoch'
    )

    model.save(os.path.join(iter_output_dir, "cnn_model.keras"))
    
    saveTestMetrics(model, x_test, y_test, num_test, iter_output_dir)

    
    new_i = getNewData(model,train_df,FEATURE_KEY,mode,num_test,sampling_size)
    print(type(new_i), type(training_i), flush=True)

    out = " ".join(list(new_i)+list(training_i))
    with open(os.path.join(iter_output_dir, 'next_index.line'), 'w') as file:
        file.write(out)

    del model
    keras.backend.clear_session()     

    
    return
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", type=str, help='Where to write the stuff')
    parser.add_argument("data_file", type=str, help='Path to file with features and lables')
    parser.add_argument("mode", type=str)
    parser.add_argument("iteration", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--initial_data", type=int, default=6000)
    parser.add_argument("--sampling_size", type=int, default =4000)
    parser.add_argument("--num_test", type=int, default=80, help="Number of times to run each sample through the node drop test")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--FEATURE_KEY", type=str, default='sequence', help="Column name(s) fo feature for model input")
    parser.add_argument("--LABEL_KEY", type=str, default='activity_bin')
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning Rate")
    args = parser.parse_args()
    
    main(**vars(args))










