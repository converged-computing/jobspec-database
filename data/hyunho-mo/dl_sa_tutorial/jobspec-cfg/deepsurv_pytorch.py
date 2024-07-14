# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import time
import sys
import argparse
import h5py
import json
import matplotlib
import matplotlib.pyplot as plt
import utils
from collections import defaultdict
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

current_dir = os.path.dirname(os.path.abspath(__file__))
configs_dir = os.path.join(current_dir, "configs")
data_dir = os.path.join(current_dir, "experiments/data/whas")
tensorboard_dir = os.path.join(current_dir, "shared/data/logs/tensorboard_")
deepsurv_dir = os.path.join(current_dir, "deepsurv")
models_dir = os.path.join(current_dir, "pytorch_models")
deepsurv_ini_path = os.path.join(configs_dir, 'deepsurv_pytorch.ini') 

import logging
import uuid
import time
import random as rnd 

# from datasets import SurvivalDataset
# from array_to_tensor import SurvivalDataset
import torch
import torch.optim as optim
import prettytable as pt
from networks import DeepSurv
from networks import NegativeLogLikelihood
from utils_torch import read_config
from utils_torch import c_index
from utils_torch import adjust_learning_rate
from utils_torch import create_logger

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest

from pytorch_dataset import Dataset_Converter


os.environ['CUDA_VISIBLE_DEVICES'] ='0'
print ("torch.cuda.is_available()", torch.cuda.is_available())
print ("Checker")

sys.path.insert(1, deepsurv_dir)
sys.path.append("/dl_sa_tutorial/deepsurv")
import deep_surv
from deepsurv_logger import TensorboardLogger

localtime   = time.localtime()
TIMESTRING  = time.strftime("%m%d%Y%M", localtime)
DURATION_COL = 'time'
EVENT_COL = 'censor'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rnd.seed(0)

def format_datasets_theano (array_train_x, array_train_y, array_train_e, array_test_x, array_test_y, array_test_e):

    train_dict = {"x":array_train_x , "t":array_train_y , "e":array_train_e }
    test_dict = {"x":array_test_x , "t":array_test_y , "e":array_test_e }
    datasets = {"train":train_dict, "test":test_dict}

    return datasets

def format_datasets_pytorch (array_train_x, array_train_y, array_train_e, array_test_x, array_test_y, array_test_e, array_val_x, array_val_y, array_val_e):

    train_dict = {"x":array_train_x , "t":array_train_y , "e":array_train_e }
    test_dict = {"x":array_test_x , "t":array_test_y , "e":array_test_e }
    val_dict = {"x":array_val_x , "t":array_val_y , "e":array_val_e }


    return train_dict, test_dict, val_dict


def evaluate_cph_model(model, dataset, bootstrap = False):
    def ci(model):
        def cph_ci(x, t, e, **kwargs):
            return concordance_index(
                event_times= t, 
                predicted_scores= -model.predict_partial_hazard(x), 
                event_observed= e,
            )
        return cph_ci

    def mse(model):
        def cph_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(-model.predict_partial_hazard(x).values)
            return ((hr_pred - hr) ** 2).mean()
        return cph_mse  

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)
    
    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics

def evaluate_theano_model(model, dataset, bootstrap = False):
    def mse(model):
        def deepsurv_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(model.predict_risk(x))
            return ((hr_pred - hr) ** 2).mean()

        return deepsurv_mse

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = model.get_concordance_index(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(model.get_concordance_index, dataset)
    
    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics

def evaluate_rsf_model(model, dataset, bootstrap = False, trt_idx = None):
    def ci(model):
        def rsf_ci(**kwargs):
            data = utils.format_dataset_to_df(kwargs, DURATION_COL, EVENT_COL, trt_idx)
            pred_test = rfSRC.predict_rfsrc(model, data)
            return 1 - pred_test.rx('err.rate')[0][-1]
        return rsf_ci

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)
    
    return metrics

def rsf_dataprep (df,x_columns, e_y_columns):
    df = df.astype({'fstat': 'bool'})
    # print (df)
    df_x = df[x_columns]
    df_rsf_y = df[e_y_columns]
    records = df_rsf_y.to_records(index=False)
    array_y = np.array(records, dtype = records.dtype.descr)

    return df_x, array_y


# def array_normalize(X):
#     ''' Performs normalizing X data. '''
#     X = (X-X.min(axis=0)) / \
#         (X.max(axis=0)-X.min(axis=0))
#     return nomalized_array

def df_to_array (df, x_columns, y_label = "lenfol", e_label = "fstat"):
    array_x = df[x_columns].to_numpy().astype(np.float32)
    array_y = df["lenfol"].to_numpy().astype(np.float32)
    array_e = df["fstat"].to_numpy().astype(np.int32)

    array_y = np.reshape(array_y, (array_y.shape[0],1))
    array_e = np.reshape(array_e, (array_e.shape[0],1))

    return array_x, array_y, array_e



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment', help="name of the experiment that is being run")
    parser.add_argument('-model', help='Model .json file to load', required=False)
    parser.add_argument('-dataset', help='.h5 File containing the train/valid/test datasets')
    parser.add_argument('--update_fn',help='Which lasagne optimizer to use (ie. sgd, adam, rmsprop)', default='sgd')
    parser.add_argument('--plot_error', action="store_true", help="If arg present, plot absolute error plots")
    parser.add_argument('--treatment_idx', default=None, type=int, help='(Optional) column index of treatment variable in dataset. If present, run treatment visualizations.')
    parser.add_argument('--results_dir', help="Output directory to save results (model weights, visualizations)", default=None)
    parser.add_argument('--weights', help='(Optional) Weights .h5 File', default=None)
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs to train for. Default: 300")
    parser.add_argument('--num_trees', default=100, type=int, help='Hyper-parameter for number of trees to grow')

    args = parser.parse_args()
    data_filename = args.dataset
    # "whas500.xls"
    data_filepath =os.path.join(data_dir, data_filename)
    
    # Import csv file and prepare arrays
    df = pd.read_excel(data_filepath)
 

    # Select X,y,e columns
    selected_columns = ["id", "age", "gender", "hr", "sysbp", "diasbp", "bmi", "cvd", "afb", "sho", "chf", "av3", "miord", "mitype", "lenfol", "fstat"]
    # selected_columns = ["id", "age", "gender", "hr", "sysbp", "diasbp", "bmi", "sho", "chf", "av3", "miord", "mitype", "lenfol", "fstat"]
    x_columns = selected_columns.copy()
    label_cols = ["id", "lenfol", "fstat"]
    e_y_columns = ["fstat", "lenfol"]

    x_columns = [ele for ele in x_columns if ele not in label_cols]
    # print (x_columns)
    df_selected = pd.DataFrame(data=df, columns=selected_columns)
    print (df_selected)

    num_instance = len(df_selected)

    random_state = 0

    # Split to train and test sets
    df_train = df_selected.sample(frac = 0.8, random_state=random_state)
    
    # Creating dataframe with
    # rest of the 50% values
    df_test = df_selected.drop(df_train.index)
    
    # print("\n80% of the given DataFrame:")
    # print(df_train)
    
    # print("\nrest 20% of the given DataFrame:")
    # print(df_test)

    # 'x': (n,d) observations (dtype = float32), 
 	# 't': (n) event times (dtype = float32),
 	# 'e': (n) event indicators (dtype = int32)

    test_x = df_test[x_columns].to_numpy().astype(np.float32)
    test_y = df_test["lenfol"].to_numpy().astype(np.float32)
    test_e = df_test["fstat"].to_numpy().astype(np.int32)
    train_x = df_train[x_columns].to_numpy().astype(np.float32)
    train_y = df_train['lenfol'].to_numpy().astype(np.float32)
    train_e = df_train["fstat"].to_numpy().astype(np.int32)
    
    # Create datasets dict from arrays
    datasets = format_datasets_theano (train_x, train_y, train_e, test_x, test_y, test_e)

    # Split to train and test sets
    df_train = df_selected.sample(frac = 0.8, random_state=random_state)
    df_test = df_selected.drop(df_train.index)
    # split to train and validation 
    df_train_frac = df_train.sample(frac = 0.8, random_state=random_state)
    df_val_frac = df_train

    test_x, test_y, test_e = df_to_array (df_test, x_columns, "lenfol", "fstat")
    train_x, train_y, train_e = df_to_array (df_train_frac, x_columns, "lenfol", "fstat")
    val_x, val_y, val_e = df_to_array (df_val_frac, x_columns, "lenfol", "fstat")

    train_dict, test_dict, val_dict = format_datasets_pytorch (train_x, train_y, train_e, test_x, test_y, test_e, val_x, val_y, val_e)



    norm_vals = {
            'mean' : datasets['train']['x'].mean(axis =0),
            'std'  : datasets['train']['x'].std(axis=0)
        }

    print("Training CPH Model")
    train_df = utils.format_dataset_to_df(datasets['train'], DURATION_COL, EVENT_COL)
    # print (train_df)

    cf = CoxPHFitter()
    results = cf.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    cf.print_summary()
    print("Train Likelihood: " + str(cf.log_likelihood_))


    if 'valid' in datasets:
        metrics = evaluate_cph_model(cf, datasets['valid'])
        print("Valid metrics: " + str(metrics))

    if 'test' in datasets:
        metrics = evaluate_cph_model(cf, datasets['test'], bootstrap=True)
        ci_cph = metrics
        print("Test metrics: " + str(metrics))
    



    # RSF evaluation
    
    ## prepare X and y
    # rsf_train_X = df_train[x_columns]
    # df_rsf_y = df_train[["fstat", "lenfol"]]
    # records = df_rsf_y.to_records(index=False)
    # rsf_train_y = np.array(records, dtype = records.dtype.descr)

    rsf_train_X, rsf_train_y = rsf_dataprep (df_train,x_columns, e_y_columns)
    rsf_test_X, rsf_test_y = rsf_dataprep (df_test,x_columns, e_y_columns)



    rsf = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=random_state)
    # n_estimators=1000, min_samples_split=5, min_samples_leaf=10, n_jobs=-1, random_state=random_state)
    rsf.fit(rsf_train_X, rsf_train_y)

    ci_rsf = rsf.score(rsf_test_X, rsf_test_y)
    print ("ci_rsf", ci_rsf)

    # Theano deepsurv evaluation
        # Train Model

    print ("################### Theano DeepSurv evaluation ###################")
    # TODO standardize location of logs + results => have them go into same directory with same UUID of experiment
    tensor_log_dir = tensorboard_dir + str(args.dataset) + "_" + str(uuid.uuid4())
    logger = TensorboardLogger("experiments.deep_surv", tensor_log_dir, update_freq = 10)
    model = deep_surv.load_model_from_json(args.model, args.weights)
    if 'valid' in datasets:
        valid_data = datasets['valid']
    else:
        valid_data = None
    # metrics = model.train(datasets['train'], valid_data, n_epochs = args.num_epochs, logger=logger,
    #     update_fn = utils.get_optimizer_from_str(args.update_fn),
    #     validation_frequency = 100)
    metrics = model.train(datasets['train'], valid_data, n_epochs = args.num_epochs, logger=logger,
        update_fn = utils.get_optimizer_from_str(args.update_fn),
        validation_frequency = 10)

    # Evaluate Model
    with open(args.model, 'r') as fp:
        json_model = fp.read()
        hyperparams = json.loads(json_model)

    train_data = datasets['train']
    if hyperparams['standardize']:
        train_data = utils.standardize_dataset(train_data, norm_vals['mean'], norm_vals['std'])

    metrics = evaluate_theano_model(model, train_data)
    print("Training metrics: " + str(metrics))
    if 'valid' in datasets:
        valid_data = datasets['valid']
        if hyperparams['standardize']:
            valid_data = utils.standardize_dataset(valid_data, norm_vals['mean'], norm_vals['std'])
            metrics = evaluate_theano_model(model, valid_data)
        print("Valid metrics: " + str(metrics))

    if 'test' in datasets:
        test_dataset = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])
        metrics = evaluate_theano_model(model, test_dataset, bootstrap=True)
        ci_theano = metrics
        print("Test metrics: " + str(metrics))

    if 'viz' in datasets:
        print("Saving Visualizations")
        save_risk_surface_visualizations(model, datasets['viz'], norm_vals = norm_vals,
            output_dir=args.results_dir, plot_error = args.plot_error, 
            experiment = args.experiment, trt_idx= args.treatment_idx)

    if 'test' in datasets and args.treatment_idx is not None:
        print("Calculating treatment recommendation survival curvs")
        # We use the test dataset because these experiments don't have a viz dataset
        save_treatment_rec_visualizations(model, test_dataset, output_dir=args.results_dir, 
            trt_idx = args.treatment_idx)

    if args.results_dir:
        _, model_str = os.path.split(args.model)
        output_file = os.path.join(args.results_dir,"models") + model_str + str(uuid.uuid4()) + ".h5"
        print("Saving model parameters to output file", output_file)
        save_model(model, output_file)





    #PyTorch deepsurv evaluation
    patience = 50
    ## Prepare dataset 




    train_dataset = Dataset_Converter(train_dict)
    validation_dataset = Dataset_Converter(val_dict)
    test_dataset = Dataset_Converter(test_dict)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=validation_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())

    ## Create a pytorch model 
    # deepsurv_ini_path
    ini_file = 'whas.ini'
    config = read_config(deepsurv_ini_path)
    print (config)
    model = DeepSurv(config['network']).to(device)
    criterion = NegativeLogLikelihood(config['network'], device).to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
        
    ## Train the network
    best_c_index = 0
    flag = 0
    stopping_index = False 
    for epoch in range(1, config['train']['epochs']+1):
        # adjusts learning rate
        
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
        # train step
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            
            risk_pred = model(X)

            # risk_pred = risk_pred.to(device)

            train_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)

            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # validation step
        model.eval()
        for X, y, e in validation_loader:
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)

            # makes predictions
            with torch.no_grad():
                risk_pred = model(X)
                # risk_pred = risk_pred.to(device)
                valid_loss = criterion(risk_pred, y, e, model)
                valid_c = c_index(-risk_pred, y, e)
                if best_c_index < valid_c:
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))

                else:
                    flag += 1         
                    print ("flag: ", flag)           
                    if flag >= patience:                        
                        print ("flag is geq patience, best_c_index_ is: ", best_c_index) 
                        stopping_index = True
                        break
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}  \n'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
        
        if stopping_index == True:
            print ("stopping index is True")
            break
    

    # test step
    model.eval()
    for X, y, e in test_loader:
        X = X.to(device)
        y = y.to(device)
        e = e.to(device)

        # makes predictions
        with torch.no_grad():
            risk_pred = model(X)
            # risk_pred = risk_pred.to(device)
            test_loss = criterion(risk_pred, y, e, model)
            test_c = c_index(-risk_pred, y, e)


    print ("torch deepsurv test ci: ", test_c)
    pytorch_ci = test_c
    ## prediction and evaluation



    print ("ci_cph", ci_cph["c_index"])
    print ("ci_rsf", ci_rsf)
    print ("ci_pytorch_val", best_c_index)
    print ("ci_pytorch_test", pytorch_ci)
    print ("ci_theano", ci_theano["c_index"])



if __name__ == '__main__':
    main()    


