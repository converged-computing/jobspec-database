import torch
import scipy
import scipy.io
import sys
from torchNMF import NMF
from torchAA import torchAA
from torchShiftAADiscTau import torchShiftAADisc
from ShiftNMFDiscTau import ShiftNMF
import numpy as np
import matplotlib.pyplot as plt

nr_components = int(sys.argv[1])
model_name = sys.argv[2]
data_name = sys.argv[3]


if data_name == "alko":
    # load data from .MAT file
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
    # Get X and Labels. Probably different for the other dataset, but i didn't check :)
    X = mat.get('xData')
    targets = mat.get('yData')
    target_labels = mat.get('yLabels')
    axis = mat.get("Axis")
    X_alko = X
if data_name == "art":
    from Artificial_shift import X
    X_art = X
if data_name == "oil":
    from helpers.data import X_clean
    X_oil = X_clean
    X = X_oil
if data_name == "urine":
    from helpers.data import X
    X_urine = X


print("starting")
print(model_name)
print(data_name)

lrs = [1, 0.1, 0.01]

nr_tests = 10
losses = np.zeros((len(lrs),nr_tests))

alpha = 1e-6

for i, lr in enumerate(lrs):
    print("learning rate:" + str(lr))
    for it in range(nr_tests):
        if model_name == "NMF":
            model = NMF(X, nr_components, lr=lr, alpha = alpha, factor=1, patience=10)
        if model_name == "AA":
            model = torchAA(X, nr_components, lr=lr, alpha = alpha, factor=1, patience=10)
        if model_name == "shiftAA":
            model = torchShiftAADisc(X, nr_components, lr=lr, alpha = alpha, factor=1, patience=10)
        if model_name == "shiftNMF":
            model = ShiftNMF(X, nr_components, lr=lr, alpha = alpha, factor=1, patience=10)
        returns = model.fit(verbose=False, return_loss=True)
        loss = returns[-1]
        losses[i,it] = loss[-1]

print(lrs)
print(np.mean(losses,axis=1).flatten())
print("all losses")
print(losses)
print("DONE")
    # plt.ylabel("average loss")
    # plt.xlabel("Learning rate")
    # plt.plot([str(lr) for lr in lrs], np.mean(losses,axis=1).flatten())
    # plt.suptitle('Categorical Plotting')
    # plt.savefig("lr_test_"+str(model_name)+"_"+str(data_name)+"_"+str(comp_nr))

np.save("./losses/"+str(data_name)+"_"+str(model_name)+"_"+str(nr_components)+"_"+"lr_test",losses)


