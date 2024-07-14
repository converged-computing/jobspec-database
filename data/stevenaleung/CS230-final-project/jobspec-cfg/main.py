import argparse
parser = argparse.ArgumentParser(description='How to run the model')
parser.add_argument("-c", "--current_run", required=True,
                    type=str,
                    help="The name of the model which will also be the name of the session's folder")
# network architecture options
parser.add_argument("-hu", "--hidden_units",
                    type=int, default=128,
                    help="The number of hidden units per layer in the RNN")
parser.add_argument("-l", "--layers",
                    type=int, default=2,
                    help="The number of layers in the RNN")
parser.add_argument("-uni", "--unidirectional",
                    action='store_true',
                    help="Use a unidirectional RNN network instead of a bidirectional network")
parser.add_argument("-ct", "--cell_type",
                    choices=['GRU', 'LSTM'], default='GRU',
                    help="Memory cell type to use in the RNN")
parser.add_argument("-bn", "--batch_norm",
                    action='store_true',
                    help="Use batch norm in every layer")
parser.add_argument("-do", "--dropout",
                    type=float, default=0,
                    help="Dropout rate")
parser.add_argument("-llo", "--layer_lock_out",
                    type=str, default = None,
                    help="Layer Lock Out --> type layers in a delimited list, e.g. \"1,2\"")
# input data options
parser.add_argument("-nsongs", "--num_songs",
                    type=int, default=349,
                    help="The total number of songs to include in the dataset")
parser.add_argument("-csv", "--datasplit_csv",
                    type=str, default=None,
                    help="The csv filename for the data split")
parser.add_argument("-sf", "--sampling_frequency",
                    type=int, default=11025,
                    help="The sampling frequency (Hz) of the audio input")
parser.add_argument("-tw", "--time_window_duration",
                    type=float, default=0.05,
                    help="The duration (s) of each time window")
parser.add_argument("-ed", "--example_duration",
                    type=float, default=4.0,
                    help="The duration (s) of each example")
# training options
parser.add_argument("-g", "--gpus",
                    type=int, default=0,
                    help="The number of GPUs to use")
parser.add_argument("-ld", "--loss_domain",
                    choices=['time', 'frequency'], default='time',
                    help="The domain in which the loss function is calculated")
parser.add_argument("-elc", "--equal_loudness_curve",
                    action='store_true',
                    help="Apply equal loudness weighting on frequency domain loss")
parser.add_argument("-bs", "--batch_size",
                    type=int, default=8,
                    help="The number of examples in each mini batch")
parser.add_argument("-lr", "--learning_rate",
                    type=float, default=0.001,
                    help="The learning rate of the RNN")
parser.add_argument("-e", "--epochs",
                    type=int, default=301,
                    help="The total number of epochs to train the RNN for")
parser.add_argument("-ste", "--starting_epoch",
                    type=int, default=0,
                    help="The starting epoch to train the RNN on")
parser.add_argument("-esi", "--epoch_save_interval",
                    type=int, default=10,
                    help="The epoch interval to save the RNN model")
parser.add_argument("-evi", "--epoch_val_interval",
                    type=int, default=10,
                    help="The epoch interval to validate the RNN model")
parser.add_argument("-eei", "--epoch_eval_interval",
                    type=int, default=10,
                    help="The epoch interval to evaluate the RNN model")
# other options
parser.add_argument("-lm", "--load_model",
                    type=str, default=None,
                    help="Folder name of model to load")
parser.add_argument("-ll", "--load_last",
                    action='store_true',
                    help="Start from last epoch")
parser.add_argument("-m", "--mode",
                    choices=['train', 'predict'], default='train',
                    help="Mode to operate model in")
# file system options
parser.add_argument("--data_dir",
                    type=str, default="./data",
                    help="Directory of datasets")
parser.add_argument("-pdd", "--predict_data_dir",
                    choices=['test_dev', 'test'], default='test',
                    help="Data used for prediction")
parser.add_argument("--runs_dir",
                    type=str, default="./runs",
                    help="The name of the model which will also be the name of the session folder")
args = parser.parse_args()


import os
from util import setup_dirs


## display input arguments
dirs = setup_dirs(args)
run_details_file = os.path.join(dirs['current_run'], 'run_details.txt')

# architecture options
architecture_options = ['hidden_units', 'layers', 'unidirectional', 'cell_type', 'batch_norm', 'dropout', 'layer_lock_out']
print('architecture options')
print('--------------------')
with open(run_details_file,'a') as file:
    print('architecture options', file=file)
    print('--------------------', file=file)
for var in architecture_options:
    print(var+' =', eval('args.'+var))
    with open(run_details_file,'a') as file:
            print(var+' =', eval('args.'+var), file=file)
print()
with open(run_details_file,'a') as file:
    print('', file=file)

# data options
data_options = ['num_songs', 'datasplit_csv', 'sampling_frequency', 
                'time_window_duration', 'example_duration']
print('data options');
print('------------')
with open(run_details_file,'a') as file:
    print('data options', file=file)
    print('------------', file=file)
for var in data_options:
    print(var+' =', eval('args.'+var))
    with open(run_details_file,'a') as file:
            print(var+' =', eval('args.'+var), file=file)
print()
with open(run_details_file,'a') as file:
    print('', file=file)

# training options
print('training options')
print('----------------')
with open(run_details_file,'a') as file:
    print('training options', file=file)
    print('----------------', file=file)
training_options = ['loss_domain', 'equal_loudness_curve', 'batch_size', 
                    'learning_rate', 'epochs', 'starting_epoch', 
                    'epoch_save_interval', 'epoch_val_interval', 
                    'epoch_eval_interval']
for var in training_options:
    print(var+' =', eval('args.'+var))
    with open(run_details_file,'a') as file:
            print(var+' =', eval('args.'+var), file=file)
print()
with open(run_details_file,'a') as file:
    print('', file=file)

# other options
print('other options')
print('-------------')
with open(run_details_file,'a') as file:
    print('other options', file=file)
    print('-------------', file=file)
other_options = ['load_model', 'load_last', 'mode', 'predict_data_dir', 
                 'data_dir', 'runs_dir']
for var in other_options:
    print(var+' =', eval('args.'+var))
    with open(run_details_file,'a') as file:
            print(var+' =', eval('args.'+var), file=file)
print()
with open(run_details_file,'a') as file:
    print('', file=file)


import numpy as np
import tensorflow as tf
from MidiNet import MidiNet
from sklearn.model_selection import train_test_split

import scipy.io
from util import setup_dirs, load_data, save_predictions, split_data, save_audio, spectrogram_loss, weighted_spectrogram_loss
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.optimizers import Nadam
import pandas as pd


from iso226 import weight_loss

# display whether gpu can be seen. if cpu & gpu are available, keras will
# automatically choose the gpu
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


def main():
    dirs = setup_dirs(args)

    num_songs = args.num_songs
    datasplit_csv = args.datasplit_csv

    # if no csv_filename provided, split data and generate a csv file
    if datasplit_csv == None:
        datasplit_dict = split_data(os.path.join(dirs['data_path'],'TPD 11kHz'), num_songs)
    else:
        df = pd.read_csv(datasplit_csv, usecols=['filename','train', 'train_dev','test'])
        datasplit_dict = df.to_dict('list')

    example_duration = args.example_duration
    time_window_duration = args.time_window_duration
    sampling_frequency = args.sampling_frequency
    loss_domain = args.loss_domain
    use_equal_loudness = args.equal_loudness_curve
    
    num_hidden_units = args.hidden_units
    num_layers = args.layers
    unidirectional_flag = args.unidirectional
    batch_norm_flag = args.batch_norm
    cell_type = args.cell_type
    batch_size = args.batch_size
    dropout = args.dropout
    learning_rate = args.learning_rate
    
    if args.layer_lock_out == None:
        layer_lock_out = []
    else:
        layer_lock_out = [int(item) for item in args.layer_lock_out.split(',')]
    layer_lock_out_mask = [(not ((l+1) in layer_lock_out)) for l in range(num_layers)] 

    num_epochs = args.epochs
    epoch_save_interval = args.epoch_save_interval

    gpus = args.gpus
    # train or evaluate the model
        # load data
    X_train, Y_train, filenames = load_data(dirs['data_path'], datasplit_dict, 'train', example_duration, time_window_duration, sampling_frequency, loss_domain, use_equal_loudness)

    # evaluate model on training and train_dev data
    X_train_dev, Y_train_dev, filenames = load_data(dirs['data_path'], datasplit_dict, 'train_dev', example_duration, time_window_duration, sampling_frequency, loss_domain, use_equal_loudness)

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = (Y_train.shape[1], Y_train.shape[2])

    # create & compile model
    if not 'model' in vars():
        with tf.device('/cpu:0'):
            elc = np.array([])
            if use_equal_loudness:
                elc = weight_loss(sampling_frequency, output_shape)
            model = MidiNet(input_shape, output_shape, loss_domain, elc, num_hidden_units, num_layers, 
                            unidirectional_flag, cell_type, batch_norm_flag, dropout, layer_lock_out_mask)
        if gpus >= 2:
            model = multi_gpu_model(model, gpus=gpus)
        if(loss_domain == "frequency"):
            opt = Nadam(clipvalue=1,lr=learning_rate)
            if(use_equal_loudness):
                model.compile(loss=weighted_spectrogram_loss, optimizer=opt)
            else:
                model.compile(loss=spectrogram_loss, optimizer=opt)
        else:
            model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    if args.load_last: # load last set of weights
        # list all the .ckpt files in a tuple (epoch, model_name)
        tree = os.listdir(dirs["weight_path"])
        files = [(int(file.split('.')[0].split('-')[1][1:]), file.split('.h5')[0]) for file in tree]
        # find the properties of the last checkpoint
        files.sort(key = lambda t: t[0])
        target_file = files[-1]
        model_epoch = target_file[0]
        model_name = target_file[1]
        model_filename = model_name + ".h5"
        print("[*] Loading " + model_filename + " and continuing from epoch " + str(model_epoch), flush=True)
        model_path = os.path.join(dirs['weight_path'], model_filename)
        model.load_weights(model_path)
        starting_epoch = int(model_epoch)+1
    else:
        starting_epoch = 0
        
    if args.mode == 'train':
        # train the model & run a checkpoint callback
        checkpoint_filename = 'model-e{epoch:03d}-loss{loss:.4f}.hdf5'
        checkpoint_filepath = os.path.join(dirs['model_path'], checkpoint_filename)
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, period=epoch_save_interval)
        
        # save weights
        cp_wt_filename = 'weights-e{epoch:03d}-loss{loss:.4f}.h5'
        cp_wt_filepath = os.path.join(dirs['weight_path'], cp_wt_filename)
        wtcheckpoint = ModelCheckpoint(cp_wt_filepath, monitor='loss', verbose=1, period=epoch_save_interval, save_weights_only=True)
        
        csv_filename = 'training_log.csv'
        csv_filepath = os.path.join(dirs['current_run'], csv_filename)
        csv_logger = CSVLogger(csv_filepath, append=True)
        callbacks_list = [checkpoint, csv_logger, wtcheckpoint]
        history_callback = model.fit(X_train, Y_train, epochs=num_epochs+starting_epoch, 
            initial_epoch=starting_epoch, batch_size=batch_size, callbacks=callbacks_list,validation_data = (X_train_dev, Y_train_dev),verbose=1)

        # save the loss history
        loss_history = history_callback.history["loss"]
        save_dict = {'loss_history': loss_history}
        filepath = os.path.join(dirs['current_run'], "loss_history.mat")
        scipy.io.savemat(filepath, save_dict)

        # save the final model
        last_epoch = history_callback.epoch[-1]
        filename = 'model-e' + str(last_epoch) + '.hdf5'
        filepath = os.path.join(dirs['model_path'], filename)
        model.save(filepath)

        Y_train_dev_pred = model.predict(X_train_dev, batch_size=batch_size)
        Y_train_pred = model.predict(X_train, batch_size=batch_size)
        
        # save audio
        print('save train audio')
        save_audio(dirs['pred_path'], 'train_dev', Y_train_dev, Y_train_dev_pred, sampling_frequency, loss_domain, use_equal_loudness)
        save_audio(dirs['pred_path'], 'train', Y_train, Y_train_pred, sampling_frequency, loss_domain, use_equal_loudness)
        
        
        # save predictions
        save_predictions(dirs['pred_path'], 'train_dev', X_train_dev, Y_train_dev, Y_train_dev_pred)
        save_predictions(dirs['pred_path'], 'train', X_train, Y_train, Y_train_pred)

        
    elif args.mode == 'predict':
        # if args.predict_data_dir == 'test_dev':
        #     data_path = dirs['test_dev_path']
        # elif args.predict_data_dir == 'test':
        #     data_path = dirs['test_path']
        X_data, Y_data, filenames = load_data(dirs['data_path'], datasplit_dict, 'test', example_duration, time_window_duration, sampling_frequency, loss_domain, use_equal_loudness)

        # evaluate model on test data
        print('[*] Making predictions', flush=True)
        Y_data_pred = model.predict(X_data, batch_size=batch_size)

          # save audio
        save_audio(dirs['pred_path'], args.predict_data_dir, Y_data, Y_data_pred, sampling_frequency, loss_domain, use_equal_loudness)
        
        # save predictions
        save_predictions(dirs['pred_path'], args.predict_data_dir, X_data, Y_data, Y_data_pred)
        print('save test audio')
      

if __name__ == '__main__':
    main()