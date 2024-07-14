import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

from scipy.stats import entropy

import gc

import coffea.hist as hist

import argparse
import ast


parser = argparse.ArgumentParser(description="Input shapes")
parser.add_argument("fromVar", type=int, help="Starting number input variable")
parser.add_argument("toVar", type=int, help="End with this input variable")
parser.add_argument("attack", help="The type of the attack, noise or fgsm")
parser.add_argument("fixRange", help="Use predefined range (yes) or just as is (no)")
parser.add_argument("evaldataset", type=str, help="Dataset used during evaluation, qcd or tt")
parser.add_argument("traindataset", type=str, help="Dataset used during training, qcd or tt")
args = parser.parse_args()

fromVar = args.fromVar
toVar = args.toVar
attack = args.attack
fixRange = args.fixRange
evaldataset = args.evaldataset
traindataset = args.traindataset

np.random.seed(0)

plt.style.use(hep.cms.style.ROOT)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


method = 0

#epsilon = 0.01

#at_epoch = 120


'''
eps = str(epsilon).replace('.','')
with open(f"/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/epsilon_{eps}/log_%d.txt" % NUM_DATASETS, "w+") as text_file:
    print(f'Do comparison plots at epoch {at_epoch} with epsilon={epsilon}', file=text_file)
print(f'Do comparison plots at epoch {at_epoch} with epsilon={epsilon}')
'''


if evaldataset == 'qcd':
    evalsuffix = '_QCD'
    NUM_DATASETS = 200 
    #NUM_DATASETS = 1
    # Old inputs, not completely clean, QCD
    scalers_file_paths = ['/work/um106329/MA/cleaned/preprocessed/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

    test_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    test_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    DeepCSV_testset_file_paths = ['/work/um106329/MA/cleaned/preprocessed/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

    '''
    # QCD clean
    scalers_file_paths = ['/hpcwork/um106329/new_march_21/scaled/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

    test_input_file_paths = ['/hpcwork/um106329/new_march_21/scaled/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    test_target_file_paths = ['/hpcwork/um106329/new_march_21/scaled/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    DeepCSV_testset_file_paths = ['/hpcwork/um106329/new_march_21/scaled/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_input_file_paths = ['/hpcwork/um106329/new_march_21/scaled/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_target_file_paths = ['/hpcwork/um106329/new_march_21/scaled/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_input_file_paths = ['/hpcwork/um106329/new_march_21/scaled/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_target_file_paths = ['/hpcwork/um106329/new_march_21/scaled/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

    # TT to Semilep clean
    scalers_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

    test_input_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    test_target_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    DeepCSV_testset_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_input_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_target_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_input_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_target_file_paths = ['/work/um106329/new_march_21/scaledTTtoSemilep/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    '''

elif evaldataset == 'tt':
    evalsuffix = '_TT'
    
    NUM_DATASETS = 49
    #NUM_DATASETS = 1
    # TT to Semilep clean with -999 instead of -1
    scalers_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

    test_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    test_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    DeepCSV_testset_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    val_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    train_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

'''

    Predictions: Without weighting
    
'''
if attack == 'fgsm':
    model0 = nn.Sequential(nn.Linear(67, 100),
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
                          nn.Softmax(dim=1))


    if traindataset == 'tt':
        all_target_file_paths_2D = [[test_target_file_paths[i],val_target_file_paths[i],train_target_file_paths[i]] for i in range(0,NUM_DATASETS)]
        all_target_file_paths = [item for sublist in all_target_file_paths_2D for item in sublist]
        flav = torch.cat(tuple(torch.load(ti) for ti in all_target_file_paths)).numpy().astype(int) + 1

        allweights = compute_class_weight(
               'balanced',
                classes=np.array([0,1,2,3]), 
                y=flav-1)
        del flav
        class_weights = torch.FloatTensor(allweights).to(device)
        del allweights
        gc.collect()

        criterion0 = nn.CrossEntropyLoss(weight=class_weights)
        del class_weights
        gc.collect()

        checkpoint0 = torch.load(f'/home/um106329/aisafety/new_march_21/models/model_all_TT_180_epochs_v10_GPU_weighted_new_49_datasets.pt', map_location=torch.device(device))

        trainsuffix = '_TT'

    elif traindataset == 'qcd':
        criterion0 = nn.CrossEntropyLoss()

        checkpoint0 = torch.load(f'/home/um106329/aisafety/models/weighted/200_full_files_120_epochs_v13_GPU_weighted_as_is.pt', map_location=torch.device(device))

        trainsuffix = '_QCD'

    model0.load_state_dict(checkpoint0["model_state_dict"])

    model0.to(device)

    #evaluate network on inputs
    model0.eval()
    #predictions_as_is = model0(test_inputs).detach().numpy()
    #print('predictions without weighting done')

'''

    Predictions: With first weighting method
    
'''
'''
# as calculated in dataset_info.ipynb
allweights1 = [0.9393934969162745, 0.9709644530642717, 0.8684253665882813, 0.2212166834311725]
class_weights1 = torch.FloatTensor(allweights1).to(device)

criterion1 = nn.CrossEntropyLoss(weight=class_weights1)



model1 = nn.Sequential(nn.Linear(67, 100),
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
                      nn.Softmax(dim=1))



checkpoint1 = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted.pt' % NUM_DATASETS, map_location=torch.device(device))
model1.load_state_dict(checkpoint1["model_state_dict"])

model1.to(device)



#evaluate network on inputs
model1.eval()
#predictions = model1(test_inputs).detach().numpy()
#print('predictions with first weighting method done')
'''


'''

    Predictions: With new weighting method
    
'''
'''
# as calculated in dataset_info.ipynb
allweights2 = [0.27580367992004956, 0.5756907770526237, 0.1270419391956182, 0.021463603831708488]      # these were the ones for QCD, n_all / n_class
class_weights2 = torch.FloatTensor(allweights2).to(device)

criterion2 = nn.CrossEntropyLoss(weight=class_weights2)



model2 = nn.Sequential(nn.Linear(67, 100),
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
                      nn.Softmax(dim=1))



checkpoint2 = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted_new.pt' % NUM_DATASETS, map_location=torch.device(device))
model2.load_state_dict(checkpoint2["model_state_dict"])

model2.to(device)




#evaluate network on inputs
model2.eval()
#predictions_new = model2(test_inputs).detach().numpy()
#print('predictions with new weighting method done')
'''







input_names = ['Jet_eta',
 'Jet_pt',
 'Jet_DeepCSV_flightDistance2dSig',
 'Jet_DeepCSV_flightDistance2dVal',
 'Jet_DeepCSV_flightDistance3dSig',
 'Jet_DeepCSV_flightDistance3dVal',
 'Jet_DeepCSV_trackDecayLenVal_0',
 'Jet_DeepCSV_trackDecayLenVal_1',
 'Jet_DeepCSV_trackDecayLenVal_2',
 'Jet_DeepCSV_trackDecayLenVal_3',
 'Jet_DeepCSV_trackDecayLenVal_4',
 'Jet_DeepCSV_trackDecayLenVal_5',
 'Jet_DeepCSV_trackDeltaR_0',
 'Jet_DeepCSV_trackDeltaR_1',
 'Jet_DeepCSV_trackDeltaR_2',
 'Jet_DeepCSV_trackDeltaR_3',
 'Jet_DeepCSV_trackDeltaR_4',
 'Jet_DeepCSV_trackDeltaR_5',
 'Jet_DeepCSV_trackEtaRel_0',
 'Jet_DeepCSV_trackEtaRel_1',
 'Jet_DeepCSV_trackEtaRel_2',
 'Jet_DeepCSV_trackEtaRel_3',
 'Jet_DeepCSV_trackJetDistVal_0',
 'Jet_DeepCSV_trackJetDistVal_1',
 'Jet_DeepCSV_trackJetDistVal_2',
 'Jet_DeepCSV_trackJetDistVal_3',
 'Jet_DeepCSV_trackJetDistVal_4',
 'Jet_DeepCSV_trackJetDistVal_5',
 'Jet_DeepCSV_trackJetPt',
 'Jet_DeepCSV_trackPtRatio_0',
 'Jet_DeepCSV_trackPtRatio_1',
 'Jet_DeepCSV_trackPtRatio_2',
 'Jet_DeepCSV_trackPtRatio_3',
 'Jet_DeepCSV_trackPtRatio_4',
 'Jet_DeepCSV_trackPtRatio_5',
 'Jet_DeepCSV_trackPtRel_0',
 'Jet_DeepCSV_trackPtRel_1',
 'Jet_DeepCSV_trackPtRel_2',
 'Jet_DeepCSV_trackPtRel_3',
 'Jet_DeepCSV_trackPtRel_4',
 'Jet_DeepCSV_trackPtRel_5',
 'Jet_DeepCSV_trackSip2dSigAboveCharm',
 'Jet_DeepCSV_trackSip2dSig_0',
 'Jet_DeepCSV_trackSip2dSig_1',
 'Jet_DeepCSV_trackSip2dSig_2',
 'Jet_DeepCSV_trackSip2dSig_3',
 'Jet_DeepCSV_trackSip2dSig_4',
 'Jet_DeepCSV_trackSip2dSig_5',
 'Jet_DeepCSV_trackSip2dValAboveCharm',
 'Jet_DeepCSV_trackSip3dSigAboveCharm',
 'Jet_DeepCSV_trackSip3dSig_0',
 'Jet_DeepCSV_trackSip3dSig_1',
 'Jet_DeepCSV_trackSip3dSig_2',
 'Jet_DeepCSV_trackSip3dSig_3',
 'Jet_DeepCSV_trackSip3dSig_4',
 'Jet_DeepCSV_trackSip3dSig_5',
 'Jet_DeepCSV_trackSip3dValAboveCharm',
 'Jet_DeepCSV_trackSumJetDeltaR',
 'Jet_DeepCSV_trackSumJetEtRatio',
 'Jet_DeepCSV_vertexCategory',
 'Jet_DeepCSV_vertexEnergyRatio',
 'Jet_DeepCSV_vertexJetDeltaR',
 'Jet_DeepCSV_vertexMass',
 'Jet_DeepCSV_jetNSecondaryVertices',
 'Jet_DeepCSV_jetNSelectedTracks',
 'Jet_DeepCSV_jetNTracksEtaRel',
 'Jet_DeepCSV_vertexNTracks',]


relative_entropies = []


display_names = ['Jet $\eta$',
                'Jet $p_T$',
                'Flight Distance 2D Sig','Flight Distance 2D Val','Flight Distance 3D Sig', 'Flight Distance 3D Val',
                'Track Decay Len Val [0]','Track Decay Len Val [1]','Track Decay Len Val [2]','Track Decay Len Val [3]','Track Decay Len Val [4]','Track Decay Len Val [5]',
                'Track $\Delta R$ [0]','Track $\Delta R$ [1]','Track $\Delta R$ [2]','Track $\Delta R$ [3]','Track $\Delta R$ [4]','Track $\Delta R$ [5]',
                'Track $\eta_{rel}$ [0]','Track $\eta_{rel}$ [1]','Track $\eta_{rel}$ [2]','Track $\eta_{rel}$ [3]',
                'Track Jet Dist Val [0]','Track Jet Dist Val [1]','Track Jet Dist Val [2]','Track Jet Dist Val [3]','Track Jet Dist Val [4]','Track Jet Dist Val [5]',
                'Track Jet $p_T$',
                'Track $p_T$ Ratio [0]','Track $p_T$ Ratio [1]','Track $p_T$ Ratio [2]','Track $p_T$ Ratio [3]','Track $p_T$ Ratio [4]','Track $p_T$ Ratio [5]',
                'Track $p_{T,rel}$ [0]','Track $p_{T,rel}$ [1]','Track $p_{T,rel}$ [2]','Track $p_{T,rel}$ [3]','Track $p_{T,rel}$ [4]','Track $p_{T,rel}$ [5]',
                'Track SIP 2D Sig Above Charm',
                'Track SIP 2D Sig [0]','Track SIP 2D Sig [1]','Track SIP 2D Sig [2]','Track SIP 2D Sig [3]','Track SIP 2D Sig [4]','Track SIP 2D Sig [5]',
                'Track SIP 2D Val Above Charm',
                'Track SIP 3D Sig Above Charm',
                'Track SIP 3D Sig [0]','Track SIP 3D Sig [1]','Track SIP 3D Sig [2]','Track SIP 3D Sig [3]','Track SIP 3D Sig [4]','Track SIP 3D Sig [5]',
                'Track SIP 3D Val Above Charm',
                'Track Sum Jet $\Delta R$','Track Sum Jet $E_T$ Ratio',
                'Vertex Category','Vertex Energy Ratio','Vertex Jet $\Delta R$','Vertex Mass',
                'Jet N Secondary Vertices','Jet N Selected Tracks','Jet N Tracks $\eta_{rel}$','Vertex N Tracks',]

def apply_noise(magn=[1],offset=[0],variable=0,minimum=None,maximum=None):
    xmagn = []
    for s in range(0, NUM_DATASETS):
        scalers = torch.load(scalers_file_paths[s])
        test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
        val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
        train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
        #test_targets =  torch.load(test_target_file_paths[s]).to(device)
        #val_targets =  torch.load(val_target_file_paths[s]).to(device)
        #train_targets =  torch.load(train_target_file_paths[s]).to(device)            
        all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))
        
        for i, m in enumerate(magn):
            noise = torch.Tensor(np.random.normal(offset,m,(len(all_inputs),67))).to(device)
            all_inputs_noise = all_inputs + noise
            if s > 0:
                xadv = scalers[variable].inverse_transform(all_inputs_noise[:][:,variable].cpu())
                integervars = [59,63,64,65,66]
                if variable in integervars:
                    xadv = np.rint(scalers[variable].inverse_transform(all_inputs[:][:,variable].cpu()))
                
                if evaldataset == 'qcd':
                    if variable in [41, 48, 49, 56]:
                        defaults = abs(scalers[variable].inverse_transform(all_inputs[:,variable].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                
                
                '''
                # as long as nothing was set to 0 manually, not really necessary
                vars_with_0_defaults = [6, 7, 8, 9, 10, 11]                 # trackDecayLenVal_0 to _5
                vars_with_0_defaults.extend([12, 13, 14, 15, 16, 17])       # trackDeltaR_0 to _5
                vars_with_0_defaults.extend([18, 19, 20, 21])               # trackEtaRel_0 to _3
                vars_with_0_defaults.extend([22, 23, 24, 25, 26, 27])       # trackJetDistVal_0 to _5
                vars_with_0_defaults.extend([29, 30, 31, 32, 33, 34])       # trackPtRatio_0 to _5
                vars_with_0_defaults.extend([35, 36, 37, 38, 39, 40])       # trackPtRel_0 to _5
                if variable in vars_with_0_defaults:
                    defaults = abs(scalers[i].inverse_transform(all_inputs[:,variable].cpu())) < 0.001   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        xadv[defaults] = all_inputs[:,variable][defaults]
                '''        
                
                '''
                    # For cleaned files (QCD or TT to Semileptonic)
                '''
                
                if variable in range(67):
                    defaults = scalers[i].inverse_transform(all_inputs[:,i].cpu()) + 900 < 0   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                        
                xadv_new = np.concatenate((xmagn[i], xadv))
                xmagn[i] = xadv_new
            else:
                xadv = scalers[variable].inverse_transform(all_inputs_noise[:][:,variable].cpu())
                integervars = [59,63,64,65,66]
                if variable in integervars:
                    xadv = np.rint(scalers[variable].inverse_transform(all_inputs[:][:,variable].cpu()))
                
                if evaldataset == 'qcd':
                    if variable in [41, 48, 49, 56]:
                        defaults = abs(scalers[variable].inverse_transform(all_inputs[:,variable].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                       
                
                '''
                # as long as nothing was set to 0 manually, not really necessary
                vars_with_0_defaults = [6, 7, 8, 9, 10, 11]                 # trackDecayLenVal_0 to _5
                vars_with_0_defaults.extend([12, 13, 14, 15, 16, 17])       # trackDeltaR_0 to _5
                vars_with_0_defaults.extend([18, 19, 20, 21])               # trackEtaRel_0 to _3
                vars_with_0_defaults.extend([22, 23, 24, 25, 26, 27])       # trackJetDistVal_0 to _5
                vars_with_0_defaults.extend([29, 30, 31, 32, 33, 34])       # trackPtRatio_0 to _5
                vars_with_0_defaults.extend([35, 36, 37, 38, 39, 40])       # trackPtRel_0 to _5
                if variable in vars_with_0_defaults:
                    defaults = abs(scalers[i].inverse_transform(all_inputs[:,variable].cpu())) < 0.001   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        xadv[defaults] = all_inputs[:,variable][defaults]
                '''        
                
                '''
                    # For cleaned files (QCD or TT to Semileptonic)
                '''
                
                if variable in range(67):
                    defaults = scalers[i].inverse_transform(all_inputs[:,variable].cpu()) + 900 < 0   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                        
                xmagn.append(xadv)
        del test_inputs
        del val_inputs
        del train_inputs
        del all_inputs
        del all_inputs_noise
        del noise
        del xadv
        gc.collect()
        
        
    if minimum is None:
        minimum = min([min(xmagn[i]) for i in range(len(magn))])
    if maximum is None:
        maximum = max([max(xmagn[i]) for i in range(len(magn))])
    bins = np.linspace(minimum+(maximum-minimum)/100/2,maximum-(maximum-minimum)/100/2,100)
    
    
    compHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Bin("prop",display_names[variable],100,minimum,maximum))
    compHist.fill(sample="raw",prop=xmagn[0])
    for si in range(1,len(magn)):
        compHist.fill(sample=f"noise $\sigma$={magn[si]}",prop=xmagn[si])
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,5],gridspec_kw={'height_ratios': [3, 1],'hspace': .3})
    hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.get_legend().remove()
    ax1.legend([f'Noise $\sigma$={magn[1]}',f'Noise $\sigma$={magn[2]}','Raw'])
    
    running_relative_entropies = []    
    for si in range(1,len(magn)):
        num = compHist[f"noise $\sigma$={magn[si]}"].sum('sample').values()[()]
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        
        '''
            Kullback-Leibler divergence with raw (denom ratio plot) and disturbed (num ratio plot) data: relative entropy
            
            S = sum(pk * log(pk / qk), axis=axis) where, in this case, p_k are the raw entries in the histogram and q_k the distorted ones
            
            This now depends on the way the bins were defined, e.g. the range and the number of bins / bin edges.
            
            Need to correct for the fact that not all bins are relevant for our studies as the defaults should be ignored / excluded from this calculation. Start to apply this
            to "unproblematic" jet variables first to avoid impact of defaults.
            EDIT: this is already covered by the defintion of minimum and maximum for the bins as specified above (implementation) / below (when calling the function),
            good to go for all the other variables as well! Expected behaviour: the variables with easily visible impact should have a higher relative entropy than those where you almost can't
            see any difference (approximately S=0), because log(ratio) will be 0 if the ratio is 1 ("no uncertainty").
        '''
        # if the distorted sample contains zero entries in a given bin, but the raw one does, the KL-divergence will be infinity (if both are 0: entropy-function assigns 0 automatically)
        # correction can be applied by replacing the "0" by "1" (this is small compared to the number of entries in total)
        # therefore one needs to check that only the distorted one is 0, the raw one is not
        
        num[(num == 0) & (denom != 0)] = 1
        
        
        entr = entropy(denom, qk=num)
        #print(f'{variable} ({input_names[variable]}):\t Noise $\sigma$={magn[si]}\t {entr}')
        running_relative_entropies.append([variable, magn[si], entr])
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        
        ax2.plot([minimum,maximum],[1,1],color='black')
        ax2.set_ylim(0,2)
        ax2.set_xlim(minimum,maximum)
        ax2.set_ylabel('Noise/raw')
    
    relative_entropies.append(running_relative_entropies)
    print(relative_entropies)
    
    sigm = ''
    for sig in magn:
        sigm = sigm + '_' + str(sig).replace('.','')
    name_var = input_names[variable]
    if fixRange == 'no':
        #fig.savefig(f'/home/um106329/aisafety/dpg21/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_no_range_spec.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_no_range_spec_oldmodel_newinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_no_range_spec_oldmodel_newQCDinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_no_range_spec_oldmodel_newTTinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/input_{variable}_{name_var}_with_noise{sigm}_no_range_spec_{NUM_DATASETS}.svg', bbox_inches='tight')
        #pass
    else:
        #fig.savefig(f'/home/um106329/aisafety/dpg21/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_specific_range.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_specific_range_oldmodel_newinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_specific_range_oldmodel_newQCDinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise/input_{variable}_{name_var}_with_noise{sigm}_specific_range_oldmodel_newTTinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/input_{variable}_{name_var}_with_noise{sigm}_specific_range_{NUM_DATASETS}.svg', bbox_inches='tight')
        #pass
    del fig, ax1, ax2
    gc.collect(2)
    

    
def fgsm_attack(epsilon=1e-1,sample=None,targets=None,reduced=True, scalers=None, model=method):
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # then we just do the forward and backwards pass as usual:
    if model == 0:
        preds = model0(xadv)
        loss = criterion0(preds, targets.long()).mean()
        model0.zero_grad()
    elif model == 1:
        preds = model1(xadv)
        loss = criterion1(preds, targets.long()).mean()
        model1.zero_grad()
    else:
        preds = model2(xadv)
        loss = criterion2(preds, targets.long()).mean()
        model2.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        #now we obtain the gradient of the input. It has the same dimensions as the tensor xadv, and it "points" in the direction of increasing loss values.
        dx = torch.sign(xadv.grad.detach())
        
        #so, we take a step in that direction!
        xadv += epsilon*torch.sign(dx)
        
        #remove the impact on selected variables. This is nessecary to avoid problems that occur otherwise in the input shapes.
        if reduced:
            #xadv[:,2] = sample[:,2]     # TagVarCSV_jetNSecondaryVertices
            xadv[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')] = sample[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')]
            #xadv[:,5] = sample[:,5]     # TagVarCSV_vertexCategory
            xadv[:,input_names.index('Jet_DeepCSV_vertexCategory')] = sample[:,input_names.index('Jet_DeepCSV_vertexCategory')]
            #xadv[:,10] = sample[:,10]   # TagVarCSV_jetNSelectedTracks
            xadv[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')] = sample[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')]
            #xadv[:,11] = sample[:,11]   # TagVarCSV_jetNTracksEtaRel
            xadv[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')] = sample[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')]
            #xadv[:,59] = sample[:,59]   # TagVarCSV_vertexNTracks
            xadv[:,input_names.index('Jet_DeepCSV_vertexNTracks')] = sample[:,input_names.index('Jet_DeepCSV_vertexNTracks')]
            #xadv[:,12:][sample[:,12:]==0] = 0   # TagVarCSVTrk_trackJetDistVal and so forth, but I have not set any variable to 0 manually during cleaning
            #xadv[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):][sample[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):] == 0] = 0
            
            '''
            for i in [41, 48, 49, 56]:
                defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                if np.sum(defaults) != 0:
                    for i in [41, 48, 49, 56]:
                        xadv[:,i][defaults] = sample[:,i][defaults]
                    break
            vars_with_0_defaults = [6, 7, 8, 9, 10, 11]                 # trackDecayLenVal_0 to _5
            vars_with_0_defaults.extend([12, 13, 14, 15, 16, 17])       # trackDeltaR_0 to _5
            vars_with_0_defaults.extend([18, 19, 20, 21])               # trackEtaRel_0 to _3
            vars_with_0_defaults.extend([22, 23, 24, 25, 26, 27])       # trackJetDistVal_0 to _5
            vars_with_0_defaults.extend([29, 30, 31, 32, 33, 34])       # trackPtRatio_0 to _5
            vars_with_0_defaults.extend([35, 36, 37, 38, 39, 40])       # trackPtRel_0 to _5
            for i in vars_with_0_defaults:
                defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu())) < 0.001   # "floating point error" --> allow some error margin
                if np.sum(defaults) != 0:
                    for i in vars_with_0_defaults:
                        xadv[:,i][defaults] = sample[:,i][defaults]
                    break
            '''
            for i in range(67):
                defaults = scalers[i].inverse_transform(sample[:,i].cpu()) + 900 < 0   # "floating point error" --> allow some error margin
                if np.sum(defaults) != 0:
                    for i in range(67):
                        xadv[:,i][defaults] = sample[:,i][defaults]
                    break
            if evaldataset ==  'qcd':
                for i in [41, 48, 49, 56]:
                    defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        for i in [41, 48, 49, 56]:
                            xadv[:,i][defaults] = sample[:,i][defaults]
        return xadv.detach()
  
       

def compare_inputs(prop=0,epsilon=0.1,minimum=None,maximum=None,reduced=True):
    xmagn = []
    for s in range(0, NUM_DATASETS):
        scalers = torch.load(scalers_file_paths[s])
        #scalers = all_scalers[s]
        test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
        val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
        train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
        test_targets =  torch.load(test_target_file_paths[s]).to(device)
        val_targets =  torch.load(val_target_file_paths[s]).to(device)
        train_targets =  torch.load(train_target_file_paths[s]).to(device)
        all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))
        all_targets = torch.cat((test_targets,val_targets,train_targets))
        #print(f'number of default -1 values for jet variables:\t{np.sum(abs(scalers[41].inverse_transform(all_inputs[:,41].cpu()) + 1.0) < 0.001)}')
        #print(f'percentage of default -1 values for jet variables:\t{np.sum(abs(scalers[41].inverse_transform(all_inputs[:,41].cpu() + 1.0)) < 0.01)/len(all_inputs[:,41].cpu())*100}%')
        
        for i, m in enumerate(epsilon):
            if s > 0:
                xadv = np.concatenate((xmagn[i], scalers[prop].inverse_transform(fgsm_attack(epsilon[i],all_inputs,all_targets,reduced=reduced, scalers=scalers, model=method)[:,prop].cpu())))
                integervars = [59,63,64,65,66]
                if prop in integervars:
                    xadv = np.rint(xadv)
                xmagn[i] = xadv
            else:
                xadv = scalers[prop].inverse_transform(fgsm_attack(epsilon[i],all_inputs,all_targets,reduced=reduced, scalers=scalers)[:,prop].cpu())
                integervars = [59,63,64,65,66]
                if prop in integervars:
                    xadv = np.rint(xadv)
                    
                xmagn.append(xadv)
        
        del scalers
        del test_inputs
        del val_inputs
        del train_inputs
        del test_targets
        del val_targets
        del train_targets
        del all_inputs
        del all_targets
        gc.collect()
    
    if minimum is None:
        minimum = min([min(xmagn[i]) for i in range(len(epsilon))])
    if maximum is None:
        maximum = max([max(xmagn[i]) for i in range(len(epsilon))])
    bins = np.linspace(minimum+(maximum-minimum)/100/2,maximum-(maximum-minimum)/100/2,100)
    
    
    compHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Bin("prop",display_names[prop],100,minimum,maximum))
    compHist.fill(sample="raw",prop=xmagn[0])
    
    for si in range(1,len(epsilon)):
        compHist.fill(sample=f"fgsm $\epsilon$={epsilon[si]}",prop=xmagn[si])
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,5],gridspec_kw={'height_ratios': [3, 1],'hspace': .3})
    hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.get_legend().remove()
    ax1.legend([f'FGSM $\epsilon$={epsilon[1]}',f'FGSM $\epsilon$={epsilon[2]}','Raw'])
    
    running_relative_entropies = []
    for si in range(1,len(epsilon)):
        num = compHist[f"fgsm $\epsilon$={epsilon[si]}"].sum('sample').values()[()]
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        '''
            Kullback-Leibler divergence with raw (denom ratio plot) and disturbed (num ratio plot) data: relative entropy
            
            As explained above
        '''
        num[(num == 0) & (denom != 0)] = 1
        entr = entropy(denom, qk=num)
        #print(f'{prop} ({input_names[prop]}):\t FGSM $\sigma$={epsilon[si]}\t {entr}')
        running_relative_entropies.append([prop, epsilon[si], entr])
        
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.plot([minimum,maximum],[1,1],color='black')    
        ax2.set_ylim(0,2)
        ax2.set_xlim(minimum,maximum)
        ax2.set_ylabel('FGSM/raw')
    
    relative_entropies.append(running_relative_entropies)
    print(relative_entropies)
        
    if method == 0:
        method_text = 'no weighting'
        filename_text = 'as_is'
    elif method == 1:
        method_text = '1 - rel. freq. weighting'
        filename_text = 'old'
    else:
        method_text = '1 / rel. freq. weighting'
        filename_text = 'new'
    if reduced == True:
        red = 'reduced'
    else:
        red = 'full'
    epsi = ''
    for eps in epsilon:
        epsi = epsi + '_' + str(eps).replace('.','')
    name_var = input_names[prop]
    if fixRange == 'no':
        #fig.savefig(f'/home/um106329/aisafety/dpg21/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_no_range_spec_{filename_text}_v2.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_no_range_spec_{filename_text}_v2_oldmodel_newinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_no_range_spec_{filename_text}_v2_oldmodel_newQCDinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_no_range_spec_{filename_text}_v2_oldmodel_newTTinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm{evalsuffix}/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_no_range_spec_{filename_text}_v2_model_{trainsuffix}_{NUM_DATASETS}.svg', bbox_inches='tight')
        #pass
    else:
        #fig.savefig(f'/home/um106329/aisafety/dpg21/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_specific_range_{filename_text}_v2.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_specific_range_{filename_text}_v2_oldmodel_newinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_specific_range_{filename_text}_v2_oldmodel_newQCDinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        #fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_specific_range_{filename_text}_v2_oldmodel_newTTinputs_{NUM_DATASETS}.svg', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/new_march_21/models/inputs_with_fgsm{evalsuffix}/input_{prop}_{name_var}_with_{red}_fgsm{epsi}_specific_range_{filename_text}_v2_model_{trainsuffix}_{NUM_DATASETS}.svg', bbox_inches='tight')
        #pass
    del fig, ax1, ax2
    gc.collect(2)
    
    
    
    
    
    
    '''
    compHist.fill(sample=f"fgsm $\epsilon$={epsilon}",prop=xadv)
    
    
    num = compHist[f"fgsm $\epsilon$={epsilon}"].sum('sample').values()[()]
    denom = compHist['raw'].sum('sample').values()[()]
    ratio = num / denom
    num_err = np.sqrt(num)
    denom_err = np.sqrt(denom)
    ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,5],gridspec_kw={'height_ratios': [3, 1],'hspace': .3})
    hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['blue','red']})
    ax1.get_legend().remove()
    ax1.legend(['FGSM','raw'])
    ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='black')
    ax2.plot([minimum,maximum],[1,1],color='black')
    ax2.set_ylim(0,2)
    ax2.set_xlim(minimum,maximum)
    ax2.set_ylabel('FGSM/raw')
    if method == 0:
        method_text = 'no weighting'
        filename_text = 'as_is'
    elif method == 1:
        method_text = '1 - rel. freq. weighting'
        filename_text = 'old'
    else:
        method_text = '1 / rel. freq. weighting'
        filename_text = 'new'
    fig.suptitle(f'During training: {method_text}, FGSM with $\epsilon={epsilon}$', fontsize=10)
    if range_given == False:
        fig.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/epsilon_{eps}/{prop}_{input_names[prop]}_reduced_{reduced}_method_{filename_text}_v3_no_range_spec_filter.png', bbox_inches='tight', dpi=300)
    else:
        fig.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/epsilon_{eps}/{prop}_{input_names[prop]}_reduced_{reduced}_method_{filename_text}_v3_specific_range_filter.png', bbox_inches='tight', dpi=300)
    del fig, ax1, ax2
    gc.collect(2)
    '''



if fixRange == 'no':
    if attack == "noise":
        for v in range(fromVar,toVar+1):
            apply_noise(magn=[0,0.05,0.1],offset=[0],variable=v,minimum=None,maximum=None)

    else:    
        for i in range(fromVar,toVar+1):
            #compare_inputs(i,epsilon=[0,0.005,0.01],minimum=None,maximum=None,reduced=False)
            #compare_inputs(i,epsilon=[0,0.005,0.01],minimum=None,maximum=None,reduced=True)
            compare_inputs(i,epsilon=[0,0.05,0.1],minimum=None,maximum=None,reduced=True)

else:
    if attack == "noise":
        magn = [0,0.05,0.1]
        if fromVar == 0:
            # Jet eta
            #apply_noise(variable=0,magn=magn,minimum=None,maximum=None)
            apply_noise(variable=0,magn=magn,minimum=None,maximum=None)
            
            # Jet pt
            #apply_noise(variable=1,magn=magn,minimum=None,maximum=1000)
            apply_noise(variable=1,magn=magn,minimum=None,maximum=250)
            
            # flightDist2DSig
            #apply_noise(variable=2,magn=magn,minimum=None,maximum=80)
            apply_noise(variable=2,magn=magn,minimum=-0.1,maximum=100)

            # flightDist2DVal
            #apply_noise(variable=3,magn=magn,minimum=None,maximum=None)
            apply_noise(variable=3,magn=magn,minimum=-0.1,maximum=2.6)

            # flightDist3DSig
            #apply_noise(variable=4,magn=magn,minimum=None,maximum=80)
            apply_noise(variable=4,magn=magn,minimum=-0.1,maximum=100)

            # flightDist3DVal
            #apply_noise(variable=5,magn=magn,minimum=None,maximum=3.5)
            apply_noise(variable=5,magn=magn,minimum=-0.1,maximum=5)


            # trackDecayLenVal
            #apply_noise(variable=6,magn=magn,minimum=-0.1,maximum=5)
            apply_noise(variable=6,magn=magn,minimum=-0.1,maximum=1)

            #apply_noise(variable=7,magn=magn,minimum=-0.1,maximum=5)
            apply_noise(variable=7,magn=magn,minimum=-0.1,maximum=1)

            #apply_noise(variable=8,magn=magn,minimum=-0.1,maximum=5)
            apply_noise(variable=8,magn=magn,minimum=-0.1,maximum=1)

            #apply_noise(variable=9,magn=magn,minimum=-0.1,maximum=5)
            apply_noise(variable=9,magn=magn,minimum=-0.1,maximum=1)

            #apply_noise(variable=10,magn=magn,minimum=-0.1,maximum=5)
            apply_noise(variable=10,magn=magn,minimum=-0.1,maximum=1)

            #apply_noise(variable=11,magn=magn,minimum=-0.1,maximum=5)
            apply_noise(variable=11,magn=magn,minimum=-0.1,maximum=1)
            
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/noise_1_kullback_leibler_{magn}.npy', kl)
            
            
        elif fromVar == 12:
            
            # trackDeltaR
            #apply_noise(variable=12,magn=magn,minimum=-0.01,maximum=0.31)
            apply_noise(variable=12,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=13,magn=magn,minimum=0,maximum=0.301)
            apply_noise(variable=13,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=14,magn=magn,minimum=0,maximum=0.5)
            apply_noise(variable=14,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=15,magn=magn,minimum=0,maximum=0.5)
            apply_noise(variable=15,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=16,magn=magn,minimum=0,maximum=0.5)
            apply_noise(variable=16,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=17,magn=magn,minimum=0,maximum=0.5)
            apply_noise(variable=17,magn=magn,minimum=-0.001,maximum=0.301)


            # trackEtaRel
            #apply_noise(variable=18,magn=magn,minimum=0,maximum=9)
            apply_noise(variable=18,magn=magn,minimum=-0.1,maximum=9)

            #apply_noise(variable=19,magn=magn,minimum=0,maximum=9)
            apply_noise(variable=19,magn=magn,minimum=-0.1,maximum=9)

            #apply_noise(variable=20,magn=magn,minimum=0,maximum=9)
            apply_noise(variable=20,magn=magn,minimum=-0.1,maximum=9)
            
            #apply_noise(variable=21,magn=magn,minimum=0,maximum=9)
            apply_noise(variable=21,magn=magn,minimum=-0.1,maximum=9)
             
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/noise_2_kullback_leibler_{magn}.npy', kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/noise_2_kullback_leibler_{magn}_onlyINFetarel3.npy', kl)

            
        elif fromVar == 22:
            
            # trackJetDistVal
            #apply_noise(variable=22,magn=magn,minimum=-0.08,maximum=0.0025)
            apply_noise(variable=22,magn=magn,minimum=-0.08,maximum=0.0025)

            #apply_noise(variable=23,magn=magn,minimum=-0.08,maximum=0.0025)
            apply_noise(variable=23,magn=magn,minimum=-0.08,maximum=0.0025)

            #apply_noise(variable=24,magn=magn,minimum=-0.1,maximum=0.01)
            apply_noise(variable=24,magn=magn,minimum=-0.08,maximum=0.0025)

            #apply_noise(variable=25,magn=magn,minimum=-0.1,maximum=0.01)
            apply_noise(variable=25,magn=magn,minimum=-0.08,maximum=0.0025)

            #apply_noise(variable=26,magn=magn,minimum=-0.1,maximum=0.01)
            apply_noise(variable=26,magn=magn,minimum=-0.08,maximum=0.0025)

            #apply_noise(variable=27,magn=magn,minimum=-0.1,maximum=0.01)
            apply_noise(variable=27,magn=magn,minimum=-0.08,maximum=0.0025)
            

            # trackJetPt
            #apply_noise(variable=28,magn=magn,minimum=None,maximum=575)
            apply_noise(variable=28,magn=magn,minimum=None,maximum=250)

            
            # trackPtRatio
            #apply_noise(variable=29,magn=magn,minimum=0,maximum=0.301)
            apply_noise(variable=29,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=30,magn=magn,minimum=0,maximum=0.301)
            apply_noise(variable=30,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=31,magn=magn,minimum=-0.05,maximum=0.4)
            apply_noise(variable=31,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=30,magn=magn,minimum=-0.05,maximum=0.4)
            apply_noise(variable=32,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=33,magn=magn,minimum=-0.05,maximum=0.4)
            apply_noise(variable=33,magn=magn,minimum=-0.001,maximum=0.301)

            #apply_noise(variable=34,magn=magn,minimum=-0.05,maximum=0.4)
            apply_noise(variable=34,magn=magn,minimum=-0.001,maximum=0.301)
             
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/noise_3_kullback_leibler_{magn}.npy', kl)
            
            
        elif fromVar == 35:
            # trackPtRel
            #apply_noise(variable=35,magn=magn,minimum=-0.1,maximum=6)
            apply_noise(variable=35,magn=magn,minimum=-0.1,maximum=3.1)

            #apply_noise(variable=36,magn=magn,minimum=-0.1,maximum=6)
            apply_noise(variable=36,magn=magn,minimum=-0.1,maximum=3.1)

            #apply_noise(variable=37,magn=magn,minimum=-0.1,maximum=6)
            apply_noise(variable=37,magn=magn,minimum=-0.1,maximum=3.1)

            #apply_noise(variable=38,magn=magn,minimum=-0.1,maximum=6)
            apply_noise(variable=38,magn=magn,minimum=-0.1,maximum=3.1)

            #apply_noise(variable=39,magn=magn,minimum=-0.1,maximum=6)
            apply_noise(variable=39,magn=magn,minimum=-0.1,maximum=3.1)

            #apply_noise(variable=40,magn=magn,minimum=-0.1,maximum=6)
            apply_noise(variable=40,magn=magn,minimum=-0.1,maximum=3.1)


            # trackSip2d (SigAboveCharm, Sig, ValAbove Charm)
            #apply_noise(variable=41,magn=magn,minimum=-5,maximum=20)
            apply_noise(variable=41,magn=magn,minimum=-5.5,maximum=5.5)

            #apply_noise(variable=42,magn=magn,minimum=-5,maximum=20)
            apply_noise(variable=42,magn=magn,minimum=-4.5,maximum=16)

            #apply_noise(variable=43,magn=magn,minimum=-5,maximum=20)
            apply_noise(variable=43,magn=magn,minimum=-5,maximum=13)
            #
            #apply_noise(variable=44,magn=magn,minimum=-20,maximum=20)
            apply_noise(variable=44,magn=magn,minimum=-5.5,maximum=10)

            #apply_noise(variable=45,magn=magn,minimum=-20,maximum=20)
            apply_noise(variable=45,magn=magn,minimum=-6,maximum=7)

            #apply_noise(variable=46,magn=magn,minimum=-20,maximum=20)
            apply_noise(variable=46,magn=magn,minimum=-6.5,maximum=4.5)

            #apply_noise(variable=47,magn=magn,minimum=-20,maximum=20)
            apply_noise(variable=47,magn=magn,minimum=-7,maximum=2)

            #apply_noise(variable=48,magn=magn,minimum=-2.1,maximum=0.1)
            apply_noise(variable=48,magn=magn,minimum=-0.06,maximum=0.06)
             
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/noise_4_kullback_leibler_{magn}.npy', kl)
            

        elif fromVar == 49:
            # trackSip3d (SigAboveCharm, Sig, ValAbove Charm)
            #apply_noise(variable=49,magn=magn,minimum=-5,maximum=20)
            apply_noise(variable=49,magn=magn,minimum=-6.5,maximum=6.5)

            #apply_noise(variable=50,magn=magn,minimum=-10,maximum=40)
            apply_noise(variable=50,magn=magn,minimum=-25,maximum=50)

            #apply_noise(variable=51,magn=magn,minimum=-10,maximum=40)
            apply_noise(variable=51,magn=magn,minimum=-25,maximum=50)

            #apply_noise(variable=52,magn=magn,minimum=-25,maximum=75)
            apply_noise(variable=52,magn=magn,minimum=-25,maximum=50)

            #apply_noise(variable=53,magn=magn,minimum=-25,maximum=75)
            apply_noise(variable=53,magn=magn,minimum=-25,maximum=50)

            #apply_noise(variable=54,magn=magn,minimum=-25,maximum=75)
            apply_noise(variable=54,magn=magn,minimum=-25,maximum=50)

            #apply_noise(variable=55,magn=magn,minimum=-25,maximum=75)
            apply_noise(variable=55,magn=magn,minimum=-25,maximum=50)

            #apply_noise(variable=56,magn=magn,minimum=-2.1,maximum=0.1)
            apply_noise(variable=56,magn=magn,minimum=-0.06,maximum=0.06)


            # trackSumJetDeltaR
            #apply_noise(variable=57,magn=magn,minimum=None,maximum=0.3)
            apply_noise(variable=57,magn=magn,minimum=-0.001,maximum=0.301)


            # trackSumJetEtRatio
            #apply_noise(variable=58,magn=magn,minimum=None,maximum=2.1)
            apply_noise(variable=58,magn=magn,minimum=None,maximum=1.4)
             
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/noise_5_kullback_leibler_{magn}.npy', kl)
            

        elif fromVar == 59:
            # vertexCat
            #apply_noise(variable=59,magn=magn,minimum=-0.1,maximum=2.1)
            apply_noise(variable=59,magn=magn,minimum=-0.6,maximum=2.6)


            # vertexEnergyRatio
            #apply_noise(variable=60,magn=magn,minimum=None,maximum=2.2)
            apply_noise(variable=60,magn=magn,minimum=0,maximum=2.5)


            # vertexJetDeltaR
            # ok
            #apply_noise(variable=61,magn=magn,minimum=None,maximum=None)
            apply_noise(variable=61,magn=magn,minimum=-0.001,maximum=0.301)


            # vertexMass
            #apply_noise(variable=62,magn=magn,minimum=None,maximum=75)
            apply_noise(variable=62,magn=magn,minimum=0,maximum=20)


            # jetNSecondaryVertices
            # ok
            #apply_noise(variable=63,magn=magn,minimum=None,maximum=None)
            apply_noise(variable=63,magn=magn,minimum=-0.5,maximum=None)


            # jetNSelectedTracks
            #apply_noise(variable=64,magn=magn,minimum=None,maximum=None)
            apply_noise(variable=64,magn=magn,minimum=-0.5,maximum=None)


            # jetNTracksEtaRel
            #apply_noise(variable=65,magn=magn,minimum=None,maximum=None)
            apply_noise(variable=65,magn=magn,minimum=-0.5,maximum=None)


            # vertexNTracks
            #apply_noise(variable=66,magn=magn,minimum=None,maximum=None)
            apply_noise(variable=66,magn=magn,minimum=-0.5,maximum=None)
             
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/noise_6_kullback_leibler_{magn}.npy', kl)
            
    else:
        #epsilon = [0,0.005,0.01]
        epsilon = [0,0.05,0.1]
        #compare_inputs(35,epsilon=[0,0.1,0.9],minimum=0,maximum=6,reduced=True)
        if fromVar == 0:
            # Jet eta
            #compare_inputs(0,epsilon,minimum=None,maximum=None,reduced=False)
            compare_inputs(0,epsilon,minimum=None,maximum=None,reduced=True)

            # Jet pt
            #compare_inputs(1,epsilon,minimum=None,maximum=1000,reduced=False)
            compare_inputs(1,epsilon,minimum=None,maximum=250,reduced=True)

            # flightDist2DSig
            #compare_inputs(2,epsilon,minimum=None,maximum=80,reduced=False)
            compare_inputs(2,epsilon,minimum=-0.1,maximum=100,reduced=True)

            # flightDist2DVal
            #compare_inputs(3,epsilon,minimum=None,maximum=None,reduced=False)
            compare_inputs(3,epsilon,minimum=-0.1,maximum=2.6,reduced=True)

            # flightDist3DSig
            #compare_inputs(4,epsilon,minimum=None,maximum=80,reduced=False)
            compare_inputs(4,epsilon,minimum=-0.1,maximum=100,reduced=True)

            # flightDist3DVal
            #compare_inputs(5,epsilon,minimum=None,maximum=3.5,reduced=False)
            compare_inputs(5,epsilon,minimum=-0.1,maximum=5,reduced=True)


            # trackDecayLenVal
            #compare_inputs(6,epsilon,minimum=-0.1,maximum=5,reduced=False)
            compare_inputs(6,epsilon,minimum=-0.1,maximum=1,reduced=True)

            #compare_inputs(7,epsilon,minimum=-0.1,maximum=5,reduced=False)
            compare_inputs(7,epsilon,minimum=-0.1,maximum=1,reduced=True)

            #compare_inputs(8,epsilon,minimum=-0.1,maximum=5,reduced=False)
            compare_inputs(8,epsilon,minimum=-0.1,maximum=1,reduced=True)

            #compare_inputs(9,epsilon,minimum=-0.1,maximum=5,reduced=False)
            compare_inputs(9,epsilon,minimum=-0.1,maximum=1,reduced=True)

            #compare_inputs(10,epsilon,minimum=-0.1,maximum=5,reduced=False)
            compare_inputs(10,epsilon,minimum=-0.1,maximum=1,reduced=True)

            #compare_inputs(11,epsilon,minimum=-0.1,maximum=5,reduced=False)
            compare_inputs(11,epsilon,minimum=-0.1,maximum=1,reduced=True)
            
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/fgsm_1_kullback_leibler_{epsilon}_model_{trainsuffix}.npy', kl)

            
        elif fromVar == 12:
            # trackDeltaR
            #compare_inputs(12,epsilon,minimum=0,maximum=0.301,reduced=False)
            compare_inputs(12,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(13,epsilon,minimum=0,maximum=0.301,reduced=False)
            compare_inputs(13,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(14,epsilon,minimum=0,maximum=0.5,reduced=False)
            compare_inputs(14,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(15,epsilon,minimum=0,maximum=0.5,reduced=False)
            compare_inputs(15,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(16,epsilon,minimum=0,maximum=0.5,reduced=False)
            compare_inputs(16,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(17,epsilon,minimum=0,maximum=0.5,reduced=False)
            compare_inputs(17,epsilon,minimum=-0.001,maximum=0.301,reduced=True)


            # trackEtaRel
            #compare_inputs(18,epsilon,minimum=0,maximum=9,reduced=False)
            compare_inputs(18,epsilon,minimum=-0.1,maximum=9,reduced=True)

            #compare_inputs(19,epsilon,minimum=0,maximum=9,reduced=False)
            compare_inputs(19,epsilon,minimum=-0.1,maximum=9,reduced=True)

            #compare_inputs(20,epsilon,minimum=0,maximum=9,reduced=False)
            compare_inputs(20,epsilon,minimum=-0.1,maximum=9,reduced=True)

            #compare_inputs(21,epsilon,minimum=0,maximum=9,reduced=False)
            compare_inputs(21,epsilon,minimum=-0.1,maximum=9,reduced=True)
             
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/fgsm_2_kullback_leibler_{epsilon}_model_{trainsuffix}.npy', kl)
            

        elif fromVar == 22:
            
            # trackJetDistVal
            #compare_inputs(22,epsilon,minimum=-0.08,maximum=0.0025,reduced=False)
            compare_inputs(22,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)

            #compare_inputs(23,epsilon,minimum=-0.08,maximum=0.0025,reduced=False)
            compare_inputs(23,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)

            #compare_inputs(24,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
            compare_inputs(24,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)

            #compare_inputs(25,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
            compare_inputs(25,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)

            #compare_inputs(26,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
            compare_inputs(26,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)

            #compare_inputs(27,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
            compare_inputs(27,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)
            

            # trackJetPt
            #compare_inputs(28,epsilon,minimum=None,maximum=575,reduced=False)
            compare_inputs(28,epsilon,minimum=None,maximum=250,reduced=True)

            
            # trackPtRatio
            #compare_inputs(29,epsilon,minimum=0,maximum=0.301,reduced=False)
            compare_inputs(29,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(30,epsilon,minimum=0,maximum=0.301,reduced=False)
            compare_inputs(30,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(31,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
            compare_inputs(31,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(30,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
            compare_inputs(32,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(33,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
            compare_inputs(33,epsilon,minimum=-0.001,maximum=0.301,reduced=True)

            #compare_inputs(34,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
            compare_inputs(34,epsilon,minimum=-0.001,maximum=0.301,reduced=True)
            
             
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/fgsm_3_kullback_leibler_{epsilon}_model_{trainsuffix}.npy', kl)
            
            
        elif fromVar == 35:
            # trackPtRel
            #compare_inputs(35,epsilon,minimum=-0.1,maximum=6,reduced=False)
            compare_inputs(35,epsilon,minimum=-0.1,maximum=3.1,reduced=True)

            #compare_inputs(36,epsilon,minimum=-0.1,maximum=6,reduced=False)
            compare_inputs(36,epsilon,minimum=-0.1,maximum=3.1,reduced=True)

            #compare_inputs(37,epsilon,minimum=-0.1,maximum=6,reduced=False)
            compare_inputs(37,epsilon,minimum=-0.1,maximum=3.1,reduced=True)

            #compare_inputs(38,epsilon,minimum=-0.1,maximum=6,reduced=False)
            compare_inputs(38,epsilon,minimum=-0.1,maximum=3.1,reduced=True)

            #compare_inputs(39,epsilon,minimum=-0.1,maximum=6,reduced=False)
            compare_inputs(39,epsilon,minimum=-0.1,maximum=3.1,reduced=True)

            #compare_inputs(40,epsilon,minimum=-0.1,maximum=6,reduced=False)
            compare_inputs(40,epsilon,minimum=-0.1,maximum=3.1,reduced=True)


            # trackSip2d (SigAboveCharm, Sig, ValAbove Charm)
            #compare_inputs(41,epsilon,minimum=-5,maximum=20,reduced=False)
            compare_inputs(41,epsilon,minimum=-5.5,maximum=5.5,reduced=True)

            #compare_inputs(42,epsilon,minimum=-5,maximum=20,reduced=False)
            compare_inputs(42,epsilon,minimum=-4.5,maximum=16,reduced=True)

            #compare_inputs(43,epsilon,minimum=-5,maximum=20,reduced=False)
            compare_inputs(43,epsilon,minimum=-5,maximum=13,reduced=True)
            #
            #compare_inputs(44,epsilon,minimum=-20,maximum=20,reduced=False)
            compare_inputs(44,epsilon,minimum=-5.5,maximum=10,reduced=True)

            #compare_inputs(45,epsilon,minimum=-20,maximum=20,reduced=False)
            compare_inputs(45,epsilon,minimum=-6,maximum=7,reduced=True)

            #compare_inputs(46,epsilon,minimum=-20,maximum=20,reduced=False)
            compare_inputs(46,epsilon,minimum=-6.5,maximum=4.5,reduced=True)

            #compare_inputs(47,epsilon,minimum=-20,maximum=20,reduced=False)
            compare_inputs(47,epsilon,minimum=-7,maximum=2,reduced=True)

            #compare_inputs(48,epsilon,minimum=-2.1,maximum=0.1,reduced=False)
            compare_inputs(48,epsilon,minimum=-0.06,maximum=0.06,reduced=True) 
            
            
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/fgsm_4_kullback_leibler_{epsilon}_model_{trainsuffix}.npy', kl)
            

        elif fromVar == 49:
            # trackSip3d (SigAboveCharm, Sig, ValAbove Charm)
            #compare_inputs(49,epsilon,minimum=-5,maximum=20,reduced=False)
            compare_inputs(49,epsilon,minimum=-6.5,maximum=6.5,reduced=True)

            #compare_inputs(50,epsilon,minimum=-10,maximum=40,reduced=False)
            compare_inputs(50,epsilon,minimum=-25,maximum=50,reduced=True)

            #compare_inputs(51,epsilon,minimum=-10,maximum=40,reduced=False)
            compare_inputs(51,epsilon,minimum=-25,maximum=50,reduced=True)

            #compare_inputs(52,epsilon,minimum=-25,maximum=75,reduced=False)
            compare_inputs(52,epsilon,minimum=-25,maximum=50,reduced=True)

            #compare_inputs(53,epsilon,minimum=-25,maximum=75,reduced=False)
            compare_inputs(53,epsilon,minimum=-25,maximum=50,reduced=True)

            #compare_inputs(54,epsilon,minimum=-25,maximum=75,reduced=False)
            compare_inputs(54,epsilon,minimum=-25,maximum=50,reduced=True)

            #compare_inputs(55,epsilon,minimum=-25,maximum=75,reduced=False)
            compare_inputs(55,epsilon,minimum=-25,maximum=50,reduced=True)

            #compare_inputs(56,epsilon,minimum=-2.1,maximum=0.1,reduced=False)
            compare_inputs(56,epsilon,minimum=-0.06,maximum=0.06,reduced=True)


            # trackSumJetDeltaR
            #compare_inputs(57,epsilon,minimum=None,maximum=0.3,reduced=False)
            compare_inputs(57,epsilon,minimum=-0.001,maximum=0.301,reduced=True)


            # trackSumJetEtRatio
            #compare_inputs(58,epsilon,minimum=None,maximum=2.1,reduced=False)
            compare_inputs(58,epsilon,minimum=None,maximum=1.4,reduced=True)
             
                
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/fgsm_5_kullback_leibler_{epsilon}_model_{trainsuffix}.npy', kl)
            

        elif fromVar == 59:
            # vertexCat
            #compare_inputs(59,epsilon,minimum=-0.1,maximum=2.1,reduced=False)
            compare_inputs(59,epsilon,minimum=-0.6,maximum=2.6,reduced=True)


            # vertexEnergyRatio
            #compare_inputs(60,epsilon,minimum=None,maximum=2.2,reduced=False)
            compare_inputs(60,epsilon,minimum=0,maximum=2.5,reduced=True)


            # vertexJetDeltaR
            # ok
            #compare_inputs(61,epsilon,minimum=None,maximum=None,reduced=False)
            compare_inputs(61,epsilon,minimum=-0.001,maximum=0.301,reduced=True)


            # vertexMass
            #compare_inputs(62,epsilon,minimum=None,maximum=75,reduced=False)
            compare_inputs(62,epsilon,minimum=0,maximum=20,reduced=True)


            # jetNSecondaryVertices
            # ok
            #compare_inputs(63,epsilon,minimum=None,maximum=None,reduced=False)
            compare_inputs(63,epsilon,minimum=-0.5,maximum=None,reduced=True)


            # jetNSelectedTracks
            #compare_inputs(64,epsilon,minimum=None,maximum=None,reduced=False)
            compare_inputs(64,epsilon,minimum=-0.5,maximum=None,reduced=True)


            # jetNTracksEtaRel
            #compare_inputs(65,epsilon,minimum=None,maximum=None,reduced=False)
            compare_inputs(65,epsilon,minimum=-0.5,maximum=None,reduced=True)


            # vertexNTracks
            #compare_inputs(66,epsilon,minimum=None,maximum=None,reduced=False)
            compare_inputs(66,epsilon,minimum=-0.5,maximum=None,reduced=True)
             
                
            kl = np.array(relative_entropies)
            print(kl)
            #np.save(f'/home/um106329/aisafety/new_march_21/models/inputs_with_noise{evalsuffix}/fgsm_6_kullback_leibler_{epsilon}_model_{trainsuffix}.npy', kl)
            