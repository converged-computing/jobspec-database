import uproot4 as uproot
import numpy as np
import awkward1 as ak

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn

from sklearn import metrics

import gc

import coffea.hist as hist



#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use(hep.cms.style.ROOT)
C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
colorcode = ['firebrick','magenta','cyan','darkgreen']
path = 'Figures/'



at_epoch = 13

NUM_DATASETS = 200




criterion = nn.CrossEntropyLoss()



model = nn.Sequential(nn.Linear(67, 100),
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



#checkpoint = torch.load(f'/home/um106329/aisafety/models/%d_full_files_{at_epoch}_epochs_v12_GPU_new.pt' % NUM_DATASETS, map_location=device)
checkpoint = torch.load(f'/home/um106329/aisafety/models/200_full_files_1_epochs_v12_GPU.pt', map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)


test_input_file_paths = ['/work/um106329/MA/preprocessed_files/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len_test)

test_target_file_paths = ['/work/um106329/MA/preprocessed_files/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = ['/work/um106329/MA/preprocessed_files/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]

test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')
DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
print('DeepCSV test done')

#evaluate network on inputs
model.eval()
predictions = model(test_inputs).detach().numpy()
print('predictions done')

jetFlavour = test_targets+1




classifierHist = hist.Hist("Jets",
                        hist.Cat("sample","sample name"),
                        hist.Cat("flavour","flavour of the jet"),
                        hist.Bin("probb","P(b)",50,0,1),
                        hist.Bin("probbb","P(bb)",50,0,1),
                        hist.Bin("probc","P(c)",50,0,1),
                        hist.Bin("probudsg","P(udsg)",50,0,1),
                     )

classifierHist.fill(sample="Classifier",flavour='b-jets',probb=predictions[:,0][jetFlavour==1],probbb=predictions[:,1][jetFlavour==1],probc=predictions[:,2][jetFlavour==1],probudsg=predictions[:,3][jetFlavour==1])
classifierHist.fill(sample="Classifier",flavour='bb-jets',probb=predictions[:,0][jetFlavour==2],probbb=predictions[:,1][jetFlavour==2],probc=predictions[:,2][jetFlavour==2],probudsg=predictions[:,3][jetFlavour==2])
classifierHist.fill(sample="Classifier",flavour='c-jets',probb=predictions[:,0][jetFlavour==3],probbb=predictions[:,1][jetFlavour==3],probc=predictions[:,2][jetFlavour==3],probudsg=predictions[:,3][jetFlavour==3])
classifierHist.fill(sample="Classifier",flavour='udsg-jets',probb=predictions[:,0][jetFlavour==4],probbb=predictions[:,1][jetFlavour==4],probc=predictions[:,2][jetFlavour==4],probudsg=predictions[:,3][jetFlavour==4])
classifierHist.fill(sample="DeepCSV",flavour='b-jets',probb=DeepCSV_testset[:,0][jetFlavour==1],probbb=DeepCSV_testset[:,1][jetFlavour==1],probc=DeepCSV_testset[:,2][jetFlavour==1],probudsg=DeepCSV_testset[:,3][jetFlavour==1])
classifierHist.fill(sample="DeepCSV",flavour='bb-jets',probb=DeepCSV_testset[:,0][jetFlavour==2],probbb=DeepCSV_testset[:,1][jetFlavour==2],probc=DeepCSV_testset[:,2][jetFlavour==2],probudsg=DeepCSV_testset[:,3][jetFlavour==2])
classifierHist.fill(sample="DeepCSV",flavour='c-jets',probb=DeepCSV_testset[:,0][jetFlavour==3],probbb=DeepCSV_testset[:,1][jetFlavour==3],probc=DeepCSV_testset[:,2][jetFlavour==3],probudsg=DeepCSV_testset[:,3][jetFlavour==3])
classifierHist.fill(sample="DeepCSV",flavour='udsg-jets',probb=DeepCSV_testset[:,0][jetFlavour==4],probbb=DeepCSV_testset[:,1][jetFlavour==4],probc=DeepCSV_testset[:,2][jetFlavour==4],probudsg=DeepCSV_testset[:,3][jetFlavour==4])



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
hist.plot1d(classifierHist['Classifier'].sum('sample','probbb','probc','probudsg'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linewidth':3})
hist.plot1d(classifierHist['Classifier'].sum('sample','probb','probc','probudsg'),overlay='flavour',ax=ax2,clear=False,line_opts={'color':colorcode,'linewidth':3})
hist.plot1d(classifierHist['Classifier'].sum('sample','probb','probbb','probudsg'),overlay='flavour',ax=ax3,clear=False,line_opts={'color':colorcode,'linewidth':3})
hist.plot1d(classifierHist['Classifier'].sum('sample','probb','probbb','probc'),overlay='flavour',ax=ax4,clear=False,line_opts={'color':colorcode,'linewidth':3})
hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
ax2.legend(loc='upper right',title='Outputs',ncol=1)
ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()
ax2.set_ylim(0, len_test)
fig.suptitle(f'Classifier and DeepCSV outputs\n After {at_epoch} epochs, evaluated on {len_test} jets')
fig.savefig(f'/home/um106329/aisafety/models/compare/after_{at_epoch}/{NUM_DATASETS}_files/hist.png', bbox_inches='tight', dpi=300)
gc.collect(2)

#plot some ROC curves
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,17],num=4)
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions[:,0])
ax1.plot(fpr,tpr)
print(f"auc for b-tagging: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,0])
ax1.plot(fpr,tpr)
ax1.legend(['Classifier','DeepCSV'])
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
ax1.set_title('b tagging')
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions[:,1])
ax2.plot(fpr,tpr)
print(f"auc for bb-tagging: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,1])
ax2.plot(fpr,tpr)
ax2.legend(['Classifier','DeepCSV'])
ax2.set_xlabel('false positive rate')
ax2.set_ylabel('true positive rate')
ax2.set_title('bb tagging')
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions[:,2])
ax3.plot(fpr,tpr)
print(f"auc for c-tagging: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,2])
ax3.plot(fpr,tpr)
ax3.legend(['Classifier','DeepCSV'])
ax3.set_xlabel('false positive rate')
ax3.set_ylabel('true positive rate')
ax3.set_title('c tagging')
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions[:,3])
ax4.plot(fpr,tpr)
print(f"auc for udsg-tagging: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,3])
ax4.plot(fpr,tpr)
ax4.legend(['Classifier','DeepCSV'])
ax4.set_xlabel('false positive rate')
ax4.set_ylabel('true positive rate')
ax4.set_title('udsg- tagging')
ax1.get_legend().remove(), ax2.get_legend().remove(), ax3.get_legend().remove()
ax4.legend(['Classifier','DeepCSV'],loc='lower right')
fig.suptitle(f'ROCs for b, bb, c and light jets\n After {at_epoch} epochs, evaluated on {len_test} jets')
fig.savefig(f'/home/um106329/aisafety/models/compare/after_{at_epoch}/{NUM_DATASETS}_files/roc.png', bbox_inches='tight', dpi=300)
gc.collect(2)


BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
del gc.garbage[:]
del test_inputs
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))
del test_targets
BvsUDSG_predictions = np.concatenate((predictions[jetFlavour==1],predictions[jetFlavour==4]))
del predictions
BvsUDSG_DeepCSV = np.concatenate((DeepCSV_testset[jetFlavour==1],DeepCSV_testset[jetFlavour==4]))
del DeepCSV_testset
gc.collect()



# plot ROC BvsUDSG

fig = plt.figure(figsize=[15,15],num=40)
fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions[:,0])
plt.plot(fpr,tpr)
print(f"auc for B vs UDSG: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0])
plt.plot(fpr,tpr)

plt.xlabel('mistag rate')
plt.ylabel('efficiency')
plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets')

plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
plt.plot([0,0.1,0.1,0,0],[0.4,0.4,0.9,0.9,0.4],'--',color='black')
plt.plot([0.1,1.01],[0.9,0.735],'--',color='grey')
plt.plot([0,0.3],[0.4,0.0],'--',color='grey')
ax = plt.axes([.37, .16, .5, .5])

fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions[:,0])
plt.plot(fpr,tpr)
fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0])
plt.plot(fpr,tpr)
plt.legend(['Classifier','DeepCSV'],loc='lower right')
plt.xlim(0,0.1)
plt.ylim(0.4,0.9)
fig.savefig(f'/home/um106329/aisafety/models/compare/after_{at_epoch}/{NUM_DATASETS}_files/roc_BvsUDSG.png', bbox_inches='tight', dpi=300)

