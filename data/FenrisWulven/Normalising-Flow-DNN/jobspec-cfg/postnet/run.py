
import torch
import torch.optim as optim
import numpy as np
import wandb
import os
import multiprocessing
from LoadDataset_class import load_Moons, load_MNIST, load_CIFAR
from PosteriorNetwork_class import NormalisingFlow, AffineCouplingLayer, Conditioner, PosteriorNetwork
from Learning_scheduler import GradualWarmupScheduler, train, init_weights
from evaluate import evaluate_model, image_show, plot_loss_acc, plot_entropy # computes test metrics

models = ['PostNet'] #['PostNet', 'Ensemble']
m = 0 # indexing for the different models
dataset_name = ['Moons', 'MNIST', 'CIFAR10'] #['CIFAR10']  #  

#for Moons, MNIST, CIFAR respectively
latent_dim = [2, 10, 6] 
num_flows = [6,6,6] # number of coupling flow layers
#training_lr = [1e-4, 1e-4, 1e-4] # start of training (end of warmup ) #Note: High LR create NaNs and Inf 
#num_epochs = [1000, 200, 200] 
#reg = [1e-5, 1e-5, 1e-5] # entropy regularisation 

warmup_steps= [1000, 1000, 1000] # batch size is larger for CIFAR10 so more steps
validation_every_steps = 50 
early_stop_patience = 20 # so after n-validations without improvement, stop training
early_stop_delta = 0.001 #in procent this is 0.1% 

num_classes = [2, 10, 10]
start_lr = 1e-9 # start of warmup
weight_decay = 1e-5  # L2 regularization strength to prevent overfitting in Adam or AdamW 
batch_size = [64,64,64]

num_params = 2 # s and t
num_hidden = [2, 3, 3] # number of hidden layers
hidden_dim = [32, 64, 64] # neurons in hidden layers
#annealing_interval = 150 # Every 10 epochs, anneal LR (warm restart)
#min_lr = 1e-6 # during cosine annealing
split = lambda x: x.chunk(2, dim=-1)

subset_percentage = 1
split_ratios = [0.6, 0.8]

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
wandb.login(key="dbe998ec8ce0708b96bb4f34fb31951d9c0eb25f")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# CIFAR flows16 
    # Reg 0, LR 0.0001
    # Reg 0.0001, LR 0.0001
    # Reg 1e-5, LR 0.0001
#DONE #Job <19727279>: Cifar, num_flows 6               Run 9
# Completed
    # Reg 0, LR 0.001
    # Reg 0, LR 0.0001
    # Reg 0.0001, LR 0.001
    # Reg 0.0001, LR 0.0001
    # Reg 1e-5,   LR 0.0001
# missing: 
#RUNNING Job 19732440:
    #CIFAR10/PostNet_Reg1e-05_LR0.001_Flows6_Warm1000_Epoch200_Wdecay1e-05_HidD64_hLay3_LD6
    #CIFAR10/PostNet_Reg0.0001_LR1e-05
    #CIFAR10/PostNet_Reg1e-05_LR1e-05
    #CIFAR10/PostNet_Reg0_LR1e-05

#DONE #Job <19727280>: Moons, num_flows [4,6,8,10,20]   Run 45 Moons/PostNet_Reg0_LR1e-05_Flows20_Warm1000_Epoch1000_Wdecay1e-05_HidD32_hLay2_LD2
#DONE #Job <19727285>: Moons, num_flows [4,6,8,10,20]   Run 45 Moons/PostNet_Reg0_LR1e-05_Flows20_Warm1000_Epoch1000_Wdecay1e-05_HidD32_hLay2_LD2
# missing:
    # Flow 20, Reg 0.0001,  LR 0.0001
    # Flow 10, Reg 0,       LR 0.0001 
    # Flow 10, Reg 0.0001,  LR 0.0001 
    # Flow 10, Reg 1e-05,   LR 0.0001
    # Flow 8,  Reg 1e-05,   LR 0.0001
    # Flow 6,  Reg 0.,      LR 0.0001 

#RUNNING #Job <19727286>: MNIST, num_flows 6            Run 9  MNIST/PostNet_Reg0.0001_LR0.001_Flows6_Warm1000_Epoch200_Wdecay1e-05_HidD64_hLay3_LD10
# Completed
    # Overwrited # Reg 0.0001, LR 0.0001  #acc 94.3% 
    # Reg 1e-5,   LR 0.001
    # Reg 1e-5,   LR 0.0001
#RUNNING #Job <19727296>: MNIST, num_flows 6            Run 9  MNIST/PostNet_Reg0.0001_LR0.001_Flows6_Warm1000_Epoch200_Wdecay1e-05_HidD64_hLay3_LD10
# Completed:
    # Reg 0.0001_LR0.001 
    # Reg 0.0001_LR0.0001           #acc 88.2%
# missing:
    # Reg 0_LR0.001


# Job interactive GPU:
# CIFAR: 
    # Reg 0_LR0.001  #batch size 256                #acc 79.6%

gamma = [0.99999, 0.9999, 0.9999] # when not using ExponentialLR scheduler
#gamma = [0.99, 0.99, 0.99] if doing manually
num_epochs = [1000, 300, 200]
num_flows = [6,8]
training_lr = [0.001, 1e-5]
reg = [1e-5]

# #Moons Missing csv
#Job <19751387>
missing_runs = [
    (0, 1e-5, 1e-5, 8), # Moons with LR 1e-5, Reg 1e-5, flows 8
    (1, 0.001, 1e-5, 6), # MNIST with LR 0.001, Reg 1e-5, flows 6
    (2, 0.001, 1e-5, 6) # CIFAR with LR 0.001, Reg 1e-5, flows 6
]
#Moons/PostNet_Reg1e-05_LR1e-05_Flows8_Warm1000_Epoch1000_Wdecay1e-05_HidD32_hLay2_LD2

count = 0
failed_runs = []
for i in range(len(dataset_name)): #reversed(range(len(dataset_name))):
    for lr in range(len(training_lr)):
        for r in range(len(reg)):
            for fl in range(len(num_flows)):

                if (i, training_lr[lr], reg[r], num_flows[fl]) not in missing_runs:
                    continue
                # if i == 0: #skip Moons 
                #     continue
                # if i == 1: # skip MNIST
                #     continue   
                # elif i == 2: # skip CIFAR
                #     continue  
                
                wandb.init(
                    project='Normalising-Flow-DNN',
                    name=f'Disk_test_{dataset_name[i]}_Reg{reg[r]}_LR{training_lr[lr]}_Flows{num_flows[fl]}_Latent{latent_dim[i]}',
                    tags=['disk'],
                    config={
                        'architecture': models[m],
                        'dataset': dataset_name[i],
                        'reg': reg[r],
                        'training_lr': training_lr[lr],
                        'num_flows': num_flows[fl],
                        'latent_dim': latent_dim[i],
                        'num_epochs': num_epochs[i],
                        'warmup_steps': warmup_steps[i],
                        'batch_size': batch_size[i],
                        'validation_every_steps': validation_every_steps,
                        'weight_decay': weight_decay,
                        'early_stop_delta': early_stop_delta,
                        'early_stop_patience': early_stop_patience,
                        'start_lr': start_lr,
                        'num_classes': num_classes,
                        'num_params': num_params,
                        'num_hidden': num_hidden[i],
                        'hidden_dim': hidden_dim[i],
                        'split': split,
                    }
                )
                
                print(f'Loading {dataset_name[i]}')
                if i==0:    
                    loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_Moons(batch_size[i], n_samples=3000, noise=0.1, split_ratios=split_ratios, seed=seed)
                elif i==1:  
                    loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_MNIST(batch_size[i], subset_percentage=None, split_ratios=split_ratios, seed=seed)
                else:       
                    loaders, N_counts, set_lengths, ood_dataset_loaders, N_ood = load_CIFAR(batch_size[i], subset_percentage=None, split_ratios=split_ratios, seed=seed)    
                print("Loader: Set length", set_lengths, "\nLoader: N_counts", N_counts)

                ### Instantiate model
                flow_models = []
                for class_label in range(num_classes[i]):
                    conditioner = Conditioner(in_dim=latent_dim[i]//2, out_dim=latent_dim[i]//2, num_hidden=num_hidden[i], hidden_dim=hidden_dim[i], num_params=num_params)
                    affine_coupling = AffineCouplingLayer(conditioner, split, latent_dim[i])
                    flows = [affine_coupling for _ in range(num_flows[fl])]
                    latent_distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim[i]).to(device), scale_tril=torch.eye(latent_dim[i]).to(device))
                    flow_model = NormalisingFlow(latent_distribution, flows).to(device)
                    #flow_model = NormalisingFlow(latent_distribution, flows).apply(init_weights).to(device) #Prevents learning???
                    flow_models.append(flow_model)
                #postnet_model = PosteriorNetwork(latent_dim[i], flow_models, N_counts['train'], num_classes[i], reg[r], dataset_name[i]).apply(init_weights).to(device)
                postnet_model = PosteriorNetwork(latent_dim[i], flow_models, N_counts['train'], num_classes[i], reg[r], dataset_name[i]).to(device)

                optimiser = optim.AdamW(postnet_model.parameters(), lr=training_lr[lr], weight_decay=weight_decay)
                warmup_scheduler = GradualWarmupScheduler(optimiser, warmup_steps=warmup_steps[i], start_lr=start_lr, end_lr=training_lr[lr])
                training_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=gamma[i]) #gamma=0.99
                #training_scheduler = optim.lr_scheduler.ConstantLR
                #training_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=annealing_interval, eta_min=min_lr, last_epoch=-1)

                model_name = f"{models[m]}_Reg{reg[r]}_LR{training_lr[lr]}_Flows{num_flows[fl]}_Warm{warmup_steps[i]}_Epoch{num_epochs[i]}_Wdecay{weight_decay}_HidD{hidden_dim[i]}_hLay{num_hidden[i]}_LD{latent_dim[i]}"
                save_model = os.path.join(dataset_name[i], model_name)  

                if torch.cuda.device_count() > 1:
                    print(f"Using {torch.cuda.device_count()} GPUs")
                    # Wrap the model for multi-GPU training
                    postnet_model = torch.nn.DataParallel(postnet_model)

                print(save_model)
                
                try: 
                    #train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses = train(postnet_model, optimiser, loaders['train'], loaders['val'], 
                    #                                        num_epochs[i], validation_every_steps, early_stop_delta, early_stop_patience, warmup_scheduler, 
                    #                                        training_scheduler, warmup_steps[i], N_counts, set_lengths, device, save_model, gamma[i])        
                    
                    
                    #plot_loss_acc(train_losses, val_losses, train_accuracies, val_accuracies, all_train_losses, dataset_name[i], model_name)
                    #print('Plotted loss')
                    
                    # For grid-search i need to do it on Validation set for both ID and OOD
                    #print("Val counts", N_counts['val'],"\nOOD Val counts", N_ood['val'])
                    print("Test counts", N_counts['test'],"\nOOD Test counts", N_ood['test'])
            
                    #val_metrics, entropy_data = evaluate_model(postnet_model, loaders['val'], ood_dataset_loaders['val'], N_counts['val'], N_ood['val'], dataset_name[i], model_name, device)
                    val_metrics, entropy_data = evaluate_model(postnet_model, loaders['test'], ood_dataset_loaders['test'], N_counts['test'], N_ood['test'], dataset_name[i], model_name, device)
                    plot_entropy(entropy_data, dataset_name[i], model_name)
                    
                    print('Printing test metrics:')
                    print(save_model)
                    for key, value in val_metrics.items():
                        if 'entropy' in key:
                            continue
                        print(f'{key}: {value}')
                
                except Exception as e:
                    print("Failed run:", save_model)
                    failed_runs.append(save_model)
                    print(f"An error occurred during training or evaluation: {e}")
                    
                    # Log the error message to Weights & Biases
                    wandb.log({"error": str(e)})

                count +=1
                print("\n Run number: ", count)
                wandb.finish()    

print("All failed runs")
for x in failed_runs:
    print(x)

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
