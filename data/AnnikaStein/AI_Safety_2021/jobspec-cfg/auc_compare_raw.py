import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn

from sklearn import metrics

import gc

import coffea.hist as hist

import time

import pickle

import argparse
import ast


parser = argparse.ArgumentParser(description="Compare AUC of different epochs")
parser.add_argument("listepochs", help="The epochs to be evaluated, specified as \"[x,y,z,...]\" ")
parser.add_argument("weighting", type=int, help="The weighting method of the training, either 0 or 2")
args = parser.parse_args()
at_epoch = ast.literal_eval(args.listepochs)
weighting_method = args.weighting


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



#at_epoch = [20,40,60,80,100,120]
#at_epoch = [i for i in range(1,121)]
#at_epoch = [20,70,120]





print(f'Evaluate at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')


'''

    Load inputs and targets
    
'''
NUM_DATASETS = 200
#NUM_DATASETS = 1   # defines the number of datasets that shall be used in the evaluation (test), if it is different from the number of files used for training

scalers_file_paths = ['/work/um106329/MA/cleaned/preprocessed/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

test_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
test_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = ['/work/um106329/MA/cleaned/preprocessed/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]


allscalers = [torch.load(scalers_file_paths[s]) for s in range(NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len(test_inputs))


test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')

jetFlavour = test_targets+1

NUM_DATASETS = 200

BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))

gc.collect()


if weighting_method == 0:

    '''

        Predictions: Without weighting

    '''
    criterion0 = nn.CrossEntropyLoss()



    model0 = [nn.Sequential(nn.Linear(67, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Linear(100, 4),
                          nn.Softmax(dim=1)) for _ in range(len(at_epoch))]



    checkpoint0 = [torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{ep}_epochs_v13_GPU_weighted_as_is.pt' % NUM_DATASETS, map_location=torch.device(device)) for ep in at_epoch]
    predictions_as_is = []
    for e in range(len(at_epoch)):
        model0[e].load_state_dict(checkpoint0[e]["model_state_dict"])

        model0[e].to(device)
        #evaluate network on inputs
        model0[e].eval()
        predictions_as_is.append(model0[e](test_inputs).detach().numpy())
        gc.collect()

    print('predictions without weighting done')
    
    BvsUDSG_predictions_as_is = [np.concatenate((predictions_as_is[e][jetFlavour==1],predictions_as_is[e][jetFlavour==4])) for e in range(len(at_epoch))]
    del predictions_as_is
    gc.collect()
else:
    '''

        Predictions: With new weighting method

    '''

    # as calculated in dataset_info.ipynb
    allweights2 = [0.27580367992004956, 0.5756907770526237, 0.1270419391956182, 0.021463603831708488]
    class_weights2 = torch.FloatTensor(allweights2).to(device)

    criterion2 = nn.CrossEntropyLoss(weight=class_weights2)



    model2 = [nn.Sequential(nn.Linear(67, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Linear(100, 4),
                          nn.Softmax(dim=1)) for _ in range(len(at_epoch))]



    checkpoint2 = [torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{ep}_epochs_v13_GPU_weighted_new.pt' % NUM_DATASETS, map_location=torch.device(device)) for ep in at_epoch]
    predictions_new = []
    for e in range(len(at_epoch)):
        model2[e].load_state_dict(checkpoint2[e]["model_state_dict"])

        model2[e].to(device)
        #evaluate network on inputs
        model2[e].eval()
        predictions_new.append(model2[e](test_inputs).detach().numpy())
        gc.collect()

    print('predictions with new weighting method done')

    BvsUDSG_predictions_new = [np.concatenate((predictions_new[e][jetFlavour==1],predictions_new[e][jetFlavour==4])) for e in range(len(at_epoch))]
    del predictions_new
    gc.collect()
    
    
del allscalers
del test_inputs
del test_targets
del jetFlavour

gc.collect()



def compare_auc_raw(method=0):
    start = time.time()
    ##### CREATING THE AUCs #####
    ### RAW ###
    auc_raw = []
    for ep in range(len(at_epoch)):
        if method == 0:
            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_as_is[ep][:,0])
        else:
            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_new[ep][:,0])
        auc_raw.append(metrics.auc(fpr,tpr))
        del fpr
        del tpr
        del thresholds
        gc.collect()
    
    if method == 0:
        with open(f'/home/um106329/aisafety/models/weighted/compare/auc/auc0_raw_{args.listepochs}.data', 'wb') as file:
            pickle.dump(auc_raw, file)
    else:
        with open(f'/home/um106329/aisafety/models/weighted/compare/auc/auc2_raw_{args.listepochs}.data', 'wb') as file:
            pickle.dump(auc_raw, file)
    
    end = time.time()
    print(f"Time to create raw AUCs: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")
    start = end
    
    gc.collect()
    
with open(f"/home/um106329/aisafety/models/weighted/compare/auc/raw_logfile_{args.weighting}_{args.listepochs}.txt", "w") as text_file:
    print(f"Will start comparing aucs now", file=text_file)        
compare_auc_raw(method=weighting_method)
