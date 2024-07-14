import sys
import os
from bioml import Data_Selector as DS
from bioml import Feature_Extraction as FE
from bioml import Align_Streams as AS
from bioml import Data_Cutter as DC
from bioml import Data_PreProcessing as DPP
from bioml import DB_connection as DBC
from bioml import Data_Loader as DL
from bioml import Data_Generator as DG

from bioml.models.functions import models_additional_functions as MAF #for step decay and grid search
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.signal as sgn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from CNN_Ameri_like_model import step_decay

import bioml.models.CNN_Ameri_like_model as MODELS
from keras import backend as K

print("GPU")
tf.test.is_gpu_available()



###VARIABLES####
exp = 'leonardo'
params = {
    'dbname': exp,
    'user': 'postgres',
    'host': '128.178.51.53',
    'port': 5432
    }
recordings = ['semgcp002_recording_0'] #, 'semgtb005_recording_1', 'semgtb005_recording_0']
channels = [['Labjack_channel1','Labjack_channel2','Labjack_channel3','Labjack_channel4','Labjack_channel5','Labjack_channel6']]
refs = [[]]
win_len = 192
step = 10

def plot_raw_data(Tables, title):
    for table in Tables:
        for recording in recordings:
            subj, cdt, nb = recording.split('_')
            cdt_nb = cdt + '_' + nb

            fig, axs = plt.subplots(6, 1, sharex=True, figsize=(16, 12))
            fig.suptitle(title)
            fig.subplots_adjust(hspace=0)
            for stream_channels in channels:
                for i, chans in enumerate(stream_channels):
                    axs[i].plot(table[table['subject_id'] == subj][table['condition'] == cdt_nb][chans])
            plt.show()

def pre_processing(Labels, Tables, recs, FSs, augment):
    print('Correct Labels')
    Labels = DPP.L_correct_Labels_6channels(Labels, 'Right')
    # Align targets with 1st stream only, it should be close to the second stream since it will be the same number of windows
    print('Align streams and targets')
    Tables, Labels = AS.HH_Align(Tables, Labels)

    ##Normalize w/ ref electrode
    # print('Normalize with reference electrode')
    # Tables = DPP.H_NormRef_Data(Tables, channels, refs)

    # Filter notch 50 + 15-500 Hz
    # print('Filter signal (Bandpass + Notch)')
    # Tables = DPP.H_filter_data(Tables, FSs, recs)

    # Cut data
    print('Create three sets: train, validation and test')
    Tables, Labels = DC.H_Cut_Data_on_percentage(Tables, Labels, val_percentage=1 / 6, test_percentage=1 / 6)

    if augment:
        # Augment data by adding channel 6 in first column and channel 1 at last column
        # (To take into account the ring-like organisation of electrodes)
        Tables[0].insert(6, 'Labjack_channel1bis', Tables[0]['Labjack_channel1'], allow_duplicates=False)
        Tables[0].insert(0, 'Labjack_channel6bis', Tables[0]['Labjack_channel6'], allow_duplicates=False)

    # Standardize (mean=0 and std=1)
    # print('Standardize channels on train dataset')
    Tables, stdscale = DPP.H_standardize_training_data(Tables, recs)

    # Standardize based on training set to avoid overfitting
    # print('Standardize channels on validation and test datasets')
    Tables = DPP.H_standardize_val_test_data(Tables, recs, stdscale)

    # Windowing
    print('Create overlapping time windows')
    Train, Val, Test = DPP.H_cut_time_windows(Tables, FSs, win_len / 1000,
                                              step / 1000)  # if epoching it will arrive here

    # Align targets on windows
    print('Align targets on windows')
    Train_Labels = AS.H_align_with_windows(Train, Labels)
    Val_Labels = AS.H_align_with_windows(Val, Labels)
    Test_Labels = AS.H_align_with_windows(Test, Labels)

    # Rescale labels to be between 0 and 10, here to avoid having recording, set, and subject
    print('Rescale targets')
    Train_Labels = DPP.L_scale_labels(Train_Labels, new_max_value=10)
    Test_Labels = DPP.L_scale_labels(Test_Labels, new_max_value=10)
    if not Val_Labels.empty:
        Val_Labels = DPP.L_scale_labels(Val_Labels, new_max_value=10)

    return Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables

def reshape_for_amerilike(Train, Val, Test):
    # reshape input for ameri-like
    # For CNN input needs to be 3d
    for i in range(len(Train[0])):
        Train[0][i] = Train[0][i].values.reshape((48, -1, 1))

    for i in range(len(Val[0])):
        Val[0][i] = Val[0][i].values.reshape((48, -1, 1))

    for i in range(len(Test[0])):
        Test[0][i] = Test[0][i].values.reshape((48, -1, 1))

    Train[0] = np.array(Train[0])
    Val[0] = np.array(Val[0])
    Test[0] = np.array(Test[0])

    return Train, Val, Test

def train_model_on_data(Train, Train_Labels, Val, Val_Labels):
    dim1, dim2, _ = Train[0][0].shape
    dim_labels = Train_Labels.shape[1]

    # Use model

    print('Train model on data')
    ################################################## TRAINING ##################################################
    # Create model
    estimator = MODELS.create_CNN_Ameri_like_model(dim1, dim2, dim_labels,
                                                   lambda_=1.675e-5,
                                                   drop_rate=0,
                                                   nbr_hid1=80,
                                                   nbr_hid2=24)

    lrate_schedule = LearningRateScheduler(step_decay)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=13, verbose=1, mode='auto', baseline=None,
                                  restore_best_weights=True)
    callbacks_list = [lrate_schedule, early_stopper]

    # Train
    print('INFO: Start Training')

    hist = estimator.fit(Train[0], Train_Labels, batch_size=128, epochs=50, shuffle=True, validation_data=(Val[0], Val_Labels),
                 callbacks=callbacks_list)
    # estimator.fit(Train[0],Train_Labels, batch_size = 128, epochs = 50, shuffle=True, callbacks = callbacks_list)

    # estimator.fit(Train,Train_Labels, batch_size = 640, epochs = 15, shuffle=True)
    return estimator, hist

def evaluate(estimator, Train, Train_Labels, Val, Val_Labels, Test, Test_Labels):
    #TODO a class for train, test, val

    # Evaluate
    print('INFO: Train evaluation')
    train_losses = estimator.evaluate(Train, Train_Labels, batch_size=600)
    print('INFO: Validation evaluation')
    Val_losses = estimator.evaluate(Val, Val_Labels, batch_size=600)
    print('INFO: Test evaluation')
    test_losses = estimator.evaluate(Test, Test_Labels, batch_size=600)


def main():
    #Connect
    print('connect to DB')
    engine, conn, cur, exp_info, subj_info = DBC.connect_DB(params)

    #Extract
    print('Extract Data and Labels')
    Tables, FSs = DL.H_extract_data(engine, recordings, channels, refs, exp_info)
    Labels, label_table_name = DL.H_load_labels(engine, cur, recordings)
    print("Data and Labels extracted")

    plot_raw_data(Tables, "raw data")
    Train_Labels, Val_Labels, Test_Labels, Train, Val, Test, Tables = pre_processing(Labels, Tables, recordings, FSs)
    plot_raw_data(Tables, "standardize training set")

    Train_Labels.values[:359, :] = 3

    Train, Val, Test = reshape_for_amerilike(Train, Val, Test)



    model, history = train_model_on_data(Train, Train_Labels, Val, Val_Labels)

    evaluate(model, Train, Train_Labels, Val, Val_Labels, Test, Test_Labels)

    with open('./trainHistoryDict2', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save('./Amerilike_6ch.h5')



