import io
import os
import time
import urllib.request
import zipfile
from statistics import mean

# Reading Data


# import src.utils
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from src.data_handeling.Preprocess import Preprocess
from src.ga.evaluate_ga import evaluate_ga
from src.utils import get_data_dim


def main():
    # Read config file

    start = time.time()
    logging.basicConfig(level="INFO")

    # Load the .yaml data
    assert len(sys.argv) == 4, "Exactly one experiment configuration file must be " \
                               "passed as a positional argument to this script. \n\n" \
                               "E.g. `python preprocess_dataset.py <path to .yaml file>`"
    with open(sys.argv[1], "r") as yaml_config_file:
        logging.info("Loading simulation settings from %s", sys.argv[1])
        experiment_config = yaml.safe_load(yaml_config_file)
    # Load the data
    data_parameters = experiment_config['data_parameters']
    ga_parameters = experiment_config['ga_parameters']
    ml_parameters = experiment_config['ml_parameters']
    logging.info("Loading data")

    np.random.seed(0)
    logging.basicConfig(level="INFO")
    logging.info("Starting GA")
    label_path = data_parameters['label_file']
    #    pdb.set_trace()
    labels = os.listdir(label_path)

    #    SMAP = ['P-1', 'S-1', 'E-1', 'E-2', 'E-5', 'E-6', 'E-7',
    #           'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'P-3',
    #          'A-2', 'A-4', 'G-2',
    #         'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9',
    #        'G-4', 'T-3', 'D-12', 'B-1', 'G-7', 'P-7',
    #       'A-5', 'A-7', 'D-13', 'A-9']

    # MSL = [
    #   'M-1', 'F-7', 'T-5', 'M-4',
    #  'M-5', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14',
    # 'T-9', 'T-8', 'D-15', 'M-7', 'F-8']
    all_files = os.listdir('./results')
    files = [file[0:-4] for file in all_files]
    path1 = os.listdir('data/ServerMachineDataset/train')

    #   MSL = [x for x in MSL if x not in files]
    #  SMAP = [x for x in SMAP if x not in files]
    SMD = [x for x in path1 if x not in files]
    SMD=sorted(SMD)

    # SMAP=list(SMAP-files)
    namee = sys.argv[2]
    if namee == 'MSL':
        Satlite = MSL
    elif namee == 'MSL':
        Satlite = MSL
    else:
        Satlite = SMD

    # pdb.set_trace()

    train_signals = Satlite

    if not os.path.exists('data'):
        response = urllib.request.urlopen(data_parameters['data_url'])
        bytes_io = io.BytesIO(response.read())

        with zipfile.ZipFile(bytes_io) as zf:
            zf.extractall()

    train_signals = os.listdir('data/train')
    test_signals = os.listdir('data/test')

    # fix random seed for reproducibility
    np.random.seed(0)

    os.makedirs('csv', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    im = 0

    # %%
    training_truth_df = pd.DataFrame()

    # df_label = df_label[df_label['chan_id'].isin(Satlite)]

    # df_label = df_label.reset_index(drop=True)
    selected_columns = ['value']
    precision_rf = []
    recall_rf = []
    Accuracy_rf = []
    F1_rf = []
    precision_lr = []
    recall_lr = []
    Accuracy_lr = []
    F1_lr = []

    precision_rnn = []
    recall_rnn = []
    Accuracy_rnn = []
    F1_rnn = []
    precision_merge = []
    recall_merge = []
    Accuracy_merge = []
    F1_merge = []
    precision_voting = []
    recall_voting = []
    Accuracy_voting = []
    F1_voting = []
    #

    #  df_label = df_label[df_label['chan_id'].isin(Satlite)]

    # df_label = df_label.reset_index(drop=True)
    Satlite1 = 'SMD'

    for name in Satlite:

        if Satlite1 == 'SMD':
            df_label = pd.read_csv(os.path.join(label_path, name), "rb")
            print(name)
            # pdb.set_trace()

            label_row = df_label  # [df_label.chan_id == name]

            x_dim = get_data_dim(Satlite1)
            #            pdb.set_trace()
            # current direcrory
            getcwd = os.getcwd()
            smd_path = getcwd + "/data/ServerMachineDataset/processed"
            f = open(os.path.join(smd_path, name[:-4] + '_train.pkl'), "rb")
            train_start = 0
            train_data = pickle.load(f).reshape((-1, x_dim))
            f.close()
            # pdb.set_trace()
            # try:
            f = open(os.path.join(smd_path, name[:-4] + '_test.pkl'), "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))
            f.close()
            # except (KeyError, FileNotFoundError):
            #   test_data = None
            # try:
            # f = open(os.path.join(smd_path, name[:-4] + "_test_label.pkl"), "rb")
            # test_label = label_data  # pickle.load(f).reshape((-1))
            # f.close()
            # except (KeyError, FileNotFoundError):
            # test_label = None
            # if do_preprocess:
            #   train_data = preprocess(train_data)
            #   test_data = preprocess(test_data)

            #            label_row=C:\Users\mmalekis\Desktop\Genetic-Algorithm-Guided-Satellite-Anomaly-Detection\data\ServerMachineDataset\test_label\machine-1-1.txt
            #
            labels = label_row
            true_indices_flat = labels
            appended_data = []

            # labels = true_indices_flat

            index = list(labels)

            # timestamp = index * 86400 + 1022819200

            anomalies = pd.DataFrame(labels)
            # appended_data.append(anomalies)

            label_data = anomalies

            signal = name
            train_np = np.loadtxt('data/ServerMachineDataset/train/' + signal, delimiter=',')
            test_np = np.loadtxt('data/ServerMachineDataset/test/' + signal, delimiter=',')
            # data = np.concatenate([train_np, test_np])
            preprocess = Preprocess(config_path=sys.argv[1])
            # test_np = preprocess.build_df(test_np, name=Satlite1)
            # train_np = preprocess.build_df(train_np, name=Satlite1)

            # pdb.set_trace()
            # data['name'] = name[:-4]
            # data['index'] = data['index'].astype(int)
            # data['day_of_week'] = data['date'].dt.dayofweek.astype(int)
            # data['hour_of_day'] = data['date'].dt.hour.astype(int)
            train_data = pd.DataFrame(train_np)
            test_data = pd.DataFrame(test_np)

            # data = data[selected_columns]
            # data.to_csv('csv/' + name[:-4] + '.csv', index=False)
            train_data.to_csv('csv/' + name[:-4] + '-train.csv', index=False)
            test_data.to_csv('csv/' + name[:-4] + '-test.csv', index=False)

        else:
            labels = label_row.anomaly_sequences[label_row.index[0]]
            # pdb.set_trace()
            labels = eval(labels)
            true_indices_grouped = [list(range(e[0], e[1] + 1)) for e in labels]
            true_indices_flat = set([i for group in true_indices_grouped for i in group])
            appended_data = []

            labels = true_indices_flat

            index = list(labels)

            # timestamp = index * 86400 + 1022819200

            anomalies = pd.DataFrame({'value': 1, 'index': index})
            appended_data.append(anomalies)

            label_data = anomalies

            signal = name
            train_np = np.load('data/train/' + signal + '.npy')
            test_np = np.load('data/test/' + signal + '.npy')
            preprocess = Preprocess(config_path=sys.argv[1])
            data = preprocess.build_df(np.concatenate([train_np, test_np]), start=0, name=Satlite1)
            data['name'] = name
            data['index'] = data['index'].astype(int)
            # data['day_of_week'] = data['date'].dt.dayofweek.astype(int)
            # data['hour_of_day'] = data['date'].dt.hour.astype(int)

            data = data[selected_columns]
            data.to_csv('csv/' + name + '.csv', index=False)
        # data = preprocess.preprocess_algorithm(data)
        # data['date'] = pd.to_datetime(data['timestamp'], unit='s')
        # data['month'] = data['date'].dt.month.astype(int)

        #  pdb.set_trace()

        dict_ensemble = evaluate_ga('/csv/' + name[:-4] + '.csv', labels, train_data, test_data, sys.argv[3])

        # pdb.set_trace()

        # anomaly_voting = voting(dict_ensemble['best_anomalies_rf'], dict_ensemble['best_anomalies_rnn'],
        #                       dict_ensemble['predicteddanomaly123'])
        precision1_rf, recall1_rf, Accuracy1_rf, F11v = dict_ensemble['best_precision_rf'], dict_ensemble[
            'best_recall_rf'], dict_ensemble['best_Accuracy_rf'], dict_ensemble['best_F1_rf']
        precision1_rnn, recall1_rnn, Accuracy1_rnn, F11r = dict_ensemble['best_precision_rnn'], dict_ensemble[
            'best_recall_rnn'], dict_ensemble['best_Accuracy_rnn'], dict_ensemble['best_F1_rnn']
        precision1_lr, recall1_lr, Accuracy1_lr, F11l = dict_ensemble['best_precision_lr'], dict_ensemble[
            'best_recall_lr'], dict_ensemble['best_Accuracy_lr'], dict_ensemble['best_F1_lr']
        precision1_merge12, recall1_merge12, Accuracy1_merge12, F11m12 = dict_ensemble['precision12'], dict_ensemble[
            'recall12'], dict_ensemble['Accuracy12'], dict_ensemble['F112']
        precision1_merge13, recall1_merge13, Accuracy1_merge13, F11m13 = dict_ensemble['precision13'], dict_ensemble[
            'recall13'], dict_ensemble['Accuracy13'], dict_ensemble['F113']
        precision1_merge23, recall1_merge23, Accuracy1_merge23, F11m23 = dict_ensemble['precision23'], dict_ensemble[
            'recall23'], dict_ensemble['Accuracy23'], dict_ensemble['F123']
        precision1_merge, recall1_merge, Accuracy1_merge, F11m = dict_ensemble['precision123'], dict_ensemble[
            'recall123'], dict_ensemble['Accuracy123'], dict_ensemble['F1123']

        # precision1_voting, recall1_voting, Accuracy1_voting, F11_voting = score(label_data, anomaly_voting,
        #    dict_ensemble['best_anomalies_rf']
        #   )
        # dict_ensemble['precision1_voting'] = precision1_voting
        # dict_ensemble['recall1_voting'] = recall1_voting
        # dict_ensemble['Accuracy1_voting'] = Accuracy1_voting
        # dict_ensemble['F1_voting'] = F11_voting

        # results=pd.DataFrame.from_dict(dict_ensemble)
        pd.DataFrame([dict_ensemble]).to_csv(  name + sys.argv[3] + '.csv', index=False)
        # pdb.set_trace()

        # precision_rf.append(precision1_rf)
        # recall_rf.append(recall1_rf)
        # Accuracy_rf.append(Accuracy1_rf)
        # F1_rf.append(F11v)
        # precision_lr.append(precision1_lr)
        # recall_lr.append(recall1_lr)
        # Accuracy_lr.append(Accuracy1_lr)
        # F1_lr.append(F11l)
        # precision_rnn.append(precision1_rnn)
        # recall_rnn.append(recall1_rnn)
        # Accuracy_rnn.append(Accuracy1_rnn)
        # F1_rnn.append(F11r)
        # pdb.set_trace()
        # precision_merge.append(precision1_merge)
        # recall_merge.append(recall1_merge)
        # Accuracy_merge.append(Accuracy1_merge)
        # F1_merge.append(F11m)
        # precision_voting.append(precision1_voting)
        # recall_voting.append(recall1_voting)
        # Accuracy_voting.append(Accuracy1_voting)
        # F1_voting.append(F11_voting)
        im += 1
        # results = pd.DataFrame()
        # results['precision_rf'] = precision_rf
        # results['recall_rf'] = recall_rf
        # results['Accuracy_rf'] = Accuracy_rf
        # results['F1_lr'] = F1_lr
        # results['precision_lr'] = precision_lr
        # results['recall_lr'] = recall_lr
        # results['Accuracy_lr'] = Accuracy_lr
        # results['F1_lr'] = F1_lr

        # results['precision_rnn'] = precision_rnn
        # results['recall_rnn'] = recall_rnn
        # results['Accuracy_rnn'] = Accuracy_rnn
        # results['F1_rnn'] = F1_rnn
        # results['precision_merge'] = precision_merge
        # results['recall_merge'] = recall_merge
        # results['Accuracy_merge'] = Accuracy_merge
        # results['F1_merge'] = F1_merge
        # results['precision_voting'] = precision_voting
        # results['recall_voting'] = recall_voting
        # results['Accuracy_voting'] = Accuracy_voting
        # results['F1_voting'] = F1_voting
        # results.to_csv('results/' + name + '.csv')
        # pdb.set_trace()
        print(name)
    recall_final_rnn = mean(dict_ensemble['rnn'])
    precision_final_rnn = mean(precision_rnn)
    F1_final_rnn = mean(F1_rnn)
    Accuracy_final_rnn = mean(Accuracy_rnn)
    recall_final_rf = mean(recall_rf)
    precision_final_rf = mean(precision_rf)
    F1_final_rf = mean(F1_rf)
    Accuracy_final_rf = mean(Accuracy_rnn)
    recall_final_merge = mean(recall_merge)
    precision_final_merge = mean(precision_merge)
    F1_final_merge = mean(F1_merge)
    Accuracy_final_merge = mean(Accuracy_merge)
    recall_final_voting = mean(recall_voting)
    precision_final_voting = mean(precision_voting)
    F1_final_voting = mean(F1_voting)
    Accuracy_final_voting = mean(Accuracy_voting)
    F1_final_lr = mean(F1_lr)
    precision_final_lr = mean(precision_lr)
    recall_final_lr = mean(recall_lr)
    Accuracy_final_lr = mean(Accuracy_lr)

    print('recall_final_rnn', recall_final_rnn)
    print('precision_final_rnn', precision_final_rnn)
    print('F1_final_rnn', F1_final_rnn)
    print('Accuracy_final_rnn', Accuracy_final_rnn)
    print('recall_final_rf', recall_final_rf)
    print('precision_final_rf', precision_final_rf)
    print('F1_final_rf', F1_final_rf)
    print('Accuracy_final_rf', Accuracy_final_rf)
    print('recall_final_merge', recall_final_merge)
    print('precision_final_merge', precision_final_merge)
    print('F1_final_merge', F1_final_merge)
    print('Accuracy_final_merge', Accuracy_final_merge)
    print('recall_final_voting', recall_final_voting)
    print('precision_final_voting', precision_final_voting)
    print('F1_final_voting', F1_final_voting)
    print('Accuracy_final_voting', Accuracy_final_voting)
    print('recall_final_lr', recall_final_lr)
    print('precision_final_lr', precision_final_lr)
    print('F1_final_lr', F1_final_lr)
    print('Accuracy_final_lr', Accuracy_final_lr)


if __name__ == '__main__':
    main()
