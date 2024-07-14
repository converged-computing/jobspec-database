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
deepsurv_ini_path = os.path.join(configs_dir, 'whas.ini') 
logdir = os.path.join(current_dir, "shared/data/logs")

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

import optunity
import configparser

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


#################
def load_logger(logdir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Print to Stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    logger.addHandler(ch)

    # Print to Log file
    fh = logging.FileHandler(os.path.join(logdir, 'log_' + str(uuid.uuid4())))
    fh.setFormatter(format)
    logger.addHandler(fh)

    return logger


def format_datasets_theano (array_train_x, array_train_y, array_train_e, array_test_x, array_test_y, array_test_e):

    train_dict = {"x":array_train_x , "t":array_train_y , "e":array_train_e }
    test_dict = {"x":array_test_x , "t":array_test_y , "e":array_test_e }
    datasets = {"train":train_dict, "test":test_dict}

    return datasets


def format_datasets_pytorch (array_train_x, array_train_y, array_train_e, array_test_x, array_test_y, array_test_e):

    train_dict = {"x":array_train_x , "t":array_train_y , "e":array_train_e }
    test_dict = {"x":array_test_x , "t":array_test_y , "e":array_test_e }

    return train_dict, test_dict

def load_box_constraints(file):
    with open(file, 'rb') as fp:
        return json.loads(fp.read())

def save_call_log(file, call_log):
    with open(file, 'wb') as fp:
        pickle.dump(call_log, fp)



def params_to_dims (num_in, num_hidden_layers, num_nodes):
    dims_list = [None] * (int(num_hidden_layers)+2) # input layer+num_hidden_layers+output layer
    for idx, layer in enumerate(dims_list):
        if idx == 0:
            dims_list[idx] = num_in
        elif idx == len(dims_list)-1:
            dims_list[idx] = 1
        else:
           dims_list[idx] = int(num_nodes)   

    return dims_list


def format_to_optunity(dataset, strata=False):
    '''
    Returns two numpy arrays (x, y) in which:
    x : is the data matrix of (num_examples, num_covariates)
    y : is a two column array containing the censor and time variables for each row in x
    Formats a dataset dictionary containing survival data with keys: 
        { 
            'x' : baseline data
            'e' : censor
            't' : event time
        }
    to a format that Optunity can use to run hyper-parameter searches on.
    '''
    x = dataset['x']
    e = dataset['e']
    t = dataset['t']
    y = np.column_stack((e, t))
    # Take the indices of censored entries as strata
    if strata:
        strata = [np.nonzero(np.logical_not(e).astype(np.int32))[0].tolist()]
    else:
        strata = None
    return (x,y,strata)


def add_single_quote_mark(s1):
    return "'{}'".format(s1)


def get_objective_function(num_in, num_epochs, logdir, update_fn = 'Adam', patience = 50):
    '''
    Returns the function for Optunity to optimize. The function returned by get_objective_function
    takes the parameters: x_train, y_train, x_test, and y_test, and any additional kwargs to 
    use as hyper-parameters.

    The objective function runs a DeepSurv model on the training data and evaluates it against the
    test set for validation. The result of the function call is the validation concordance index 
    (which Optunity tries to optimize)
    '''
    
    def format_to_pytorch_deepsurv(x, y):
        '''
        process x,y
        return pytorch data 

        '''
        array_x = x
        array_e = y[:,0].astype(np.int32)
        array_t = y[:,1].astype(np.float32)

        array_t = np.reshape(array_t, (array_t.shape[0],1))
        array_e = np.reshape(array_e, (array_e.shape[0],1))

        dict = {"x":array_x , "t":array_t , "e":array_e }

        return dict


    def get_hyperparams(params, num_in, update_fn, patience):
        # additional necessary haperparams not specified in the json file 
        hyperparams = {
            'activation': 'ReLU', # or 'SELU'
            'norm': None,
            'optimizer': update_fn,
            'patience': patience
        }
        # @TODO add default parameters and only take necessary args from params
        # protect from params including some other key
        if 'num_nodes' in params and 'num_hidden_layers' in params:

            params['dims'] = [None] * (int(params['num_hidden_layers'])+2) # input layer+num_hidden_layers+output layer
            for idx, layer in enumerate(params['dims']):
                if idx == 0:
                    params['dims'][idx] = num_in
                elif idx == len(params['dims'])-1:
                    params['dims'][idx] = 1
                else:
                    params['dims'][idx] = int(params['num_nodes'])      

        if 'learning_rate' in params:
            params['learning_rate'] = 10 ** params['learning_rate']

        hyperparams.update(params)
        return hyperparams


    def train_deepsurv(x_train, y_train, x_test, y_test,  
        **kwargs):
        # Standardize the datasets
        train_mean = x_train.mean(axis = 0)
        train_std = x_train.std(axis = 0)

        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std


        train_dict = format_to_pytorch_deepsurv(x_train, y_train)
        valid_dict = format_to_pytorch_deepsurv(x_test, y_test)


        train_dataset = Dataset_Converter(train_dict)
        valid_dataset = Dataset_Converter(valid_dict)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_dataset.__len__())
        validation_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=valid_dataset.__len__())

        hyperparams = get_hyperparams(kwargs, x_train.shape[1], update_fn, patience)

        # Set up Tensorboard loggers
        # TODO improve the model_id for Tensorboard to better partition runs
        model_id = str(hash(str(hyperparams)))
        run_id = model_id + '_' + str(uuid.uuid4())
        logger = TensorboardLogger('hyperparam_search', 
            os.path.join(logdir,"tensor_logs", model_id, run_id))

        # Generate an individual (Pytorch DeepSurv network for a given hyperparameters setting)
        model = DeepSurv(hyperparams).to(device)
        criterion = NegativeLogLikelihood(hyperparams, device).to(device)
        optimizer = eval('optim.{}'.format(hyperparams['optimizer']))(
            model.parameters(), lr=hyperparams['learning_rate'])

        # Train and validation (calculate CI on validation set after training the network)
            ## Train the network
        ini_file = 'temp.ini'
        best_c_index = 0
        flag = 0
        for epoch in range(1, num_epochs+1):
            # adjusts learning rate
            lr = adjust_learning_rate(optimizer, epoch,
                                    hyperparams['learning_rate'],
                                    hyperparams['lr_decay_rate'])
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
            
            # test step
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
                        if flag >= hyperparams['patience']:
                            print ("early stopping at %s" %epoch)
                            main_logger.info('Run id: %s | %s | C-Index: %f | Train Loss %f' % (run_id, str(hyperparams), best_c_index, train_loss))
                            return best_c_index
                            # print ("flag is geq patience, best_c_index_ is: ", best_c_index) 
            # notes that, train loader and valid loader both have one batch!!!
            # print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}  \n'.format(
            #     epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
          
       
        # result (which is return of the function) corresponds to Calculated CI on validation set 
        result = best_c_index
        main_logger.info('Run id: %s | %s | C-Index: %f | Train Loss %f' % (run_id, str(hyperparams), result, train_loss))
        main_logger.info('C-Index: %f | Train Loss %f' % (result, train_loss))

        return result

    return train_deepsurv

########################



def main():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('-logdir', help='Directory for tensorboard logs')
    parser.add_argument('-dataset', help='Dataset to load')
    parser.add_argument('-box', help='Filename to box constraints dictionary pickle file')
    parser.add_argument('--num_evals', help='Number of models to test', type=int, default=50)
    parser.add_argument('--update_fn',help='Which lasagne optimizer to use (ie. sgd, adam, rmsprop)', default='Adam')
    parser.add_argument('--num_epochs',type=int, help='Number of epochs to train', default=300)
    parser.add_argument('--num_folds', type=int, help='Number of folds to cross-validate', default=5)

    # box path: ~/hmo/dl_sa_tutorial/hyperparam_search/box_constraints.0.json
    # "python", "-u", "/hyperparam_search.py", \
    # "/shared/logs", \
    # "gaussian", \
    # "/box_constraints.0.json", \
    # "50", \
    # "--update_fn", "adam", \
    # "--num_epochs", "500", \
    # "--num_fold", "3" ]

    args = parser.parse_args()
    data_filename = args.dataset

    # Load WHAS500 and prepare dataset

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

    num_in = int(train_x.shape[1])    
   
    NUM_EPOCHS = args.num_epochs
    NUM_FOLDS = args.num_folds

    global main_logger
    main_logger = load_logger(logdir)
    main_logger.debug('Parameters: ' + str(args))  
    main_logger.debug('Loading dataset: ' + args.dataset)
    train_dataset = datasets["train"]

    x, y, strata = format_to_optunity(train_dataset)
    # print ("x", x)
    # print ("strata", strata)

    box_constraints = load_box_constraints(args.box)
    main_logger.debug('Box Constraints: ' + str(box_constraints))

    # Define objective function
    opt_fxn = get_objective_function(num_in, NUM_EPOCHS, logdir, 
        args.update_fn)
    
    # print ("opt_fxn", opt_fxn)

    opt_fxn = optunity.cross_validated(x=x, y=y, num_folds=NUM_FOLDS,
        strata=strata)(opt_fxn)

    # Maximize the objective function by training individuals
    main_logger.debug('Maximizing C-Index. Num_iterations: %d' % args.num_evals)
    opt_params, call_log, _ = optunity.maximize(opt_fxn, num_evals=args.num_evals,
        solver_name='sobol',
        **box_constraints)
    

    print ("opt_params", opt_params)
    # opt_params {'learning_rate': -2.9730957031250003, 
    # 'num_nodes': 28.62427734375, 
    # 'num_hidden_layers': 1.29810546875, 
    # 'lr_decay_rate': 0.000887685546875, 
    # 'l2_reg': 0.1273920898437499, 
    # 'drop': 0.1953759765625}
    # Export the optimal hyperparameter values to ini file

    print('Create new INI file and save config to the ini file')



    input_config_file = os.path.join(configs_dir, "deepsurv_pytorch.ini")
    config = configparser.ConfigParser(allow_no_value=True)
    # config.optionxform = str
    config.add_section('train')
    config['train']['epochs'] = str(NUM_EPOCHS)
    config['train']['learning_rate'] = str(10 ** opt_params['learning_rate'])
    config['train']['lr_decay_rate'] = str(opt_params['lr_decay_rate'])
    config['train']['optimizer'] = add_single_quote_mark(args.update_fn)

    dims_list = params_to_dims(num_in, opt_params['num_hidden_layers'], opt_params['num_nodes'])

    config.add_section('network')
    config['network']['drop'] = str(opt_params['drop'])
    config['network']['norm'] = str(True)
    config['network']['dims'] = str(dims_list)
    config['network']['activation'] =  add_single_quote_mark('ReLU')  
    config['network']['l2_reg'] = str(opt_params['l2_reg'])

    with open(input_config_file, 'w') as configfile:
        config.write(configfile)

    print ("Config file of the optimization is saved")



if __name__ == '__main__':
    main()    


