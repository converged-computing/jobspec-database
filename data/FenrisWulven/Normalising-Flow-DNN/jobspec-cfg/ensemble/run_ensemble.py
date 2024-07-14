
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import multiprocessing

import sys
get_cwd = os.getcwd()
root_dir = get_cwd #os.path.dirname(get_cwd)
sys.path.append(root_dir)  # add parent directory

from postnet.LoadDataset_class import load_Moons, load_MNIST, load_CIFAR
from postnet.Encoder_CIFAR import Encoder_CIFAR, BasicBlock
from postnet.Encoder_MNIST import Encoder_MNIST
from postnet.Encoder_Moons import Encoder_Moons
from postnet.Learning_scheduler import GradualWarmupScheduler
from postnet.evaluate import evaluate_model, image_show, plot_loss_acc, plot_entropy # computes test metrics
from train_ensemble import train_ensemble
from evaluate_ensemble import evaluate_ensemble

models = ['Ensemble'] #['PostNet', 'Ensemble']
m = 0 # indexing for the different models
dataset_name = ['Moons', 'MNIST', 'CIFAR10'] #['CIFAR10']  #  

#for Moons, MNIST, CIFAR respectively
#training_lr = [1e-4, 1e-4, 1e-4] # start of training (end of warmup ) #Note: High LR create NaNs and Inf 
#num_epochs = [1000, 200, 200] 
#reg = [1e-5, 1e-5, 1e-5] # entropy regularisation 

warmup_steps= [1000, 1000, 1000] # batch size is larger for CIFAR10 so more steps
validation_every_steps = 50 
early_stop_patience = 20 # so after n-validations without improvement, stop training
early_stop_delta = 0.0001 #in procent this is 0.1% 

num_classes = [2, 10, 10]
start_lr = 1e-9 # start of warmup
weight_decay = 1e-5  # L2 regularization strength to prevent overfitting in Adam or AdamW 
batch_size = [64,64,64]
subset_percentage = 1
split_ratios = [0.6, 0.8]

seed = 123
ens_seeds = [0,1,2,3,4,5,6,7,8,9]
torch.manual_seed(seed)
np.random.seed(seed)
wandb.login(key="dbe998ec8ce0708b96bb4f34fb31951d9c0eb25f")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

gamma = [0.99999, 0.9999, 0.9999]#[0.999999, 0.99999, 0.99999] # if doing manually
num_epochs = [1000, 500, 500]
training_lr = [0.01, 0.001]#[1e-2, 1e-3, 1e-4, 1e-5]  
num_ensemble_models = 10

#Job ensem_19741553 Moons, MNIST for LR=[1e-2, 1e-3, 1e-4, 1e-5] and NumEns = 10
# Stopped at: CIFAR10/Ensemble_Ens4LR0.001_Warm1000_Epoch500_Wdecay1e-05

# Job Ensemb_cif_ 19743664 is doing all of the CIFAR ensembles training

missing_runs = [
    (0, 0.001), # Moons with LR 0.001
    (1, 0.001), # MNIST with LR 0.001
    (2, 0.01) # CIFAR with LR 0.01
]

count = 0
for i in range(len(dataset_name)): #reversed(range(len(dataset_name))):
    for lr in range(len(training_lr)):
        # if i == 0: #skip Moons 
        #     continue
        # if i == 1: # skip MNIST
        #     continue   
        # elif i == 2: # skip CIFAR
        #     continue   
        if (i, training_lr[lr]) not in missing_runs:
            continue
        
        # Load dataset once
        print(f'Loading {dataset_name[i]}')
        if i==0:    
            loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_Moons(batch_size[i], n_samples=3000, noise=0.1, split_ratios=split_ratios, seed=seed)
            ensemble = Encoder_Moons(num_classes[i]).to(device)
        elif i==1:  
            loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_MNIST(batch_size[i], subset_percentage=None, split_ratios=split_ratios, seed=seed)
            ensemble = Encoder_MNIST(num_classes[i]).to(device)
        else:       
            loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_CIFAR(batch_size[i], subset_percentage=None, split_ratios=split_ratios, seed=seed) 
            ensemble = Encoder_CIFAR(block=BasicBlock, num_blocks=[2, 2, 2, 2], output_dim=num_classes[i]).to(device)
        print("Loader: Set length", set_lengths, "\nLoader: N_counts", N_counts)
        
        # Instantiate models
        ensemble_models = []
        optimisers = []
        for n in range(num_ensemble_models):
            if i==0:    
                ensemble = Encoder_Moons(num_classes[i]).to(device)
            elif i==1:  
                ensemble = Encoder_MNIST(num_classes[i]).to(device)
            else:       
                ensemble = Encoder_CIFAR(block=BasicBlock, num_blocks=[2, 2, 2, 2], output_dim=num_classes[i]).to(device)
            
            ensemble_models.append(ensemble)
            optimiser = optim.AdamW(ensemble.parameters(), lr=training_lr[lr], weight_decay=weight_decay)
            optimisers.append(optimiser)

        criterion = nn.CrossEntropyLoss()
        model_names = []
        ensemble_number = 0
        for model, optimiser in zip(ensemble_models, optimisers):

            wandb.init(
            project='Normalising-Flow-DNN',
            name=f'Ensemble_{dataset_name[i]}_LR{training_lr[lr]}_Ens{ensemble_number}_NumEns{num_ensemble_models}',
            tags=['Ensemble'],
            config={
                'architecture': models[m],
                'dataset': dataset_name[i],
                'training_lr': training_lr[lr],
                'num_ensemble_models': num_ensemble_models,
                'num_epochs': num_epochs[i],
                'warmup_steps': warmup_steps[i],
                'batch_size': batch_size[i],
                'validation_every_steps': validation_every_steps,
                'weight_decay': weight_decay,
                'early_stop_delta': early_stop_delta,
                'early_stop_patience': early_stop_patience,
                'start_lr': start_lr,})

            model_name = f"{models[m]}_Ens{ensemble_number}LR{training_lr[lr]}_Warm{warmup_steps[i]}_Epoch{num_epochs[i]}_Wdecay{weight_decay}"
            model_names.append(model_name)
            save_model = os.path.join(dataset_name[i], model_name)
            print(save_model)

            #set seed for each model
            print('Ensemble number before seed', ensemble_number)
            torch.manual_seed(ens_seeds[ensemble_number])
            np.random.seed(ens_seeds[ensemble_number])
            print("Seed: ", ens_seeds[ensemble_number])

            warmup_scheduler = GradualWarmupScheduler(optimiser, warmup_steps=warmup_steps[i], start_lr=start_lr, end_lr=training_lr[lr])
            # try:
                # train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses = train_ensemble(model, optimiser, loaders['train'], loaders['val'], 
                #                             num_epochs[i],  validation_every_steps, early_stop_delta, early_stop_patience, warmup_scheduler,
                #                             warmup_steps[i], ensemble_number, criterion, set_lengths, device, save_model, gamma[i])
                # plot_loss_acc(train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses, dataset_name[i], model_name)
                # print('Plotted loss')
            # except Exception as e:
            #     print(f"An error occurred during training: {e}")
            #     # Log the error message to Weights & Biases
            #     wandb.log({"error": str(e)})
            ensemble_number += 1
            print('Finished training model number: ', ensemble_number)
             
            print(save_model)
            
            #ensemble_number += 1 ##### Either here or above - choose
            wandb.finish()

        ensemble_name = f"{models[m]}_LR{training_lr[lr]}_Warm{warmup_steps[i]}_Epoch{num_epochs[i]}_Wdecay{weight_decay}"
        #f"{models[m]}_Ens{ensemble_number}LR{training_lr[lr]}_Warm{warmup_steps[i]}_Epoch{num_epochs[i]}_Wdecay{weight_decay}"
        print("ensemble name", ensemble_name)
        #val_metrics, entropy_data = evaluate_ensemble(ensemble_models, loaders['val'], ood_dataset_loaders['val'], dataset_name[i], model_names, ensemble_name, device)
        val_metrics, entropy_data = evaluate_ensemble(ensemble_models, loaders['test'], ood_dataset_loaders['test'], dataset_name[i], model_names, ensemble_name, device)
        print("entropy data \n",entropy_data)
        
        try:
            plot_entropy(entropy_data, dataset_name[i], ensemble_name)
        except Exception as e:
            print(f"An error occurred during plotting entropy: {e}")

        #test_metrics, entropy_data = evaluate_model(postnet_model, loaders['test'], ood_dataset_loaders, N_counts['test'], N_ood, dataset_name[i], model_name, device)
        
        print('Printing validation metrics:')
        print(save_model)
        for key, value in val_metrics.items():
            if 'entropy' in key:
                continue
            print(f'{key}: {value}')

        count +=1
        print("\n Run number: ", count, '\n\n')
        #wandb.finish()    

# import multiprocessing
# from itertools import product
# def train_model(params):
#     lr, reg, ... = params
#     # Set up data loaders, model, optimizer, etc.
#     # Optionally use DataParallel if multiple GPUs are available
#     # Train the model
#     # Save results, models, etc.

# if __name__ == "__main__":
#     hyperparameters = {
#         'learning_rate': [1e-4, 5e-5, 1e-5],
#         'regularization': [1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 0],
#         # ... other hyperparameters
#     }

#     all_params = list(product(*hyperparameters.values()))
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
#         pool.map(train_model, all_params)
