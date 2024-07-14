import h5py
import os

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.data.sampler import SubsetRandomSampler

# Path
encoder_path = "Conv2D_encoder_best_Gadi.pth"
decoder_path = "Conv2D_decoder_best_Gadi.pth"
model_path = "FNN_best_Gadi.pth"

#Parameters
batch_size = 16
accurate_loss_baseline = 1e-6

'''
path = "/scratch/kr97/xh7958/comp4560/solutions"


# Temperature for the two consecutive timestamp
temperature_fields = []
  
# Read text File  
def read_text_file(file_path):
    with h5py.File(file_path, 'r') as f:
        temperature_fields.append(f['temperature'][:])
        
        
# Iterate through all file
for file in os.listdir(path):
    file_path = f"{path}/{file}"
  
    # call read text file function
    read_text_file(file_path)
    #print(f"{file_path} is finished reading")

temperature_fields = np.asarray(temperature_fields)
temperature_fields_input = temperature_fields[:,:99,:,:].reshape(-1,201,401)
temperature_fields_output = temperature_fields[:,1:,:,:].reshape(-1,201,401)
'''
temperature_fields = np.load('/scratch/kr97/xh7958/comp4560/solutions_standard.npy')
temperature_fields_input = temperature_fields[:,:99,:,:].reshape(-1,201,401)
temperature_fields_output = temperature_fields[:,1:,:,:].reshape(-1,201,401)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
print("Current device is ",device)

# make results determinstic
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential( # 1x201x401 => 6x23x45
            nn.Conv2d(1, 3, stride=(3, 3), kernel_size=(5, 5), padding=2),
            nn.Tanh(),
            nn.Conv2d(3, 6, stride=(3, 3), kernel_size=(5, 5), padding=2),
            nn.Tanh(),
        )
        
    def forward(self, x):
        out=self.encoder(x)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential( # 6x23x45 => 201x401
            nn.ConvTranspose2d(6, 3, stride=(3, 3), kernel_size=(5, 5), padding=(2,2)),
            nn.Tanh(),
            nn.ConvTranspose2d(3, 1, stride=(3, 3), kernel_size=(5, 5), padding=(1,0)),
        )
        

    def forward(self, x):
        out=self.decoder(x)
        return out
    
encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

print("Encoder and Decoder loaded!")


# Customised Dataset class
class KMNIST(Dataset):
    
    def __init__(self, input_data, output_data):
        self.input = input_data
        self.output = output_data
        
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_item = self.input[index]
        output_item = self.output[index]
        
        return input_item, output_item
    
temperature_dataset = KMNIST(
    temperature_fields_input,
    temperature_fields_output
)


testingAndValidation_split = 0.2
validation_split = 0.1

# Creating data indices for training, testing and validation splits
# Reference: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
temperature_dataset_size = len(temperature_dataset)
temperature_indices = list(range(temperature_dataset_size))

temperature_training_testing_split = int(np.floor(testingAndValidation_split * temperature_dataset_size))
temperature_testing_validation_split = int(np.floor(validation_split * temperature_dataset_size))

np.random.shuffle(temperature_indices)
temperature_train_indices, temperature_val_indices ,temperature_test_indices = temperature_indices[temperature_training_testing_split:], temperature_indices[:temperature_testing_validation_split], temperature_indices[temperature_testing_validation_split:temperature_training_testing_split] 

# Creating data samplers
temperature_train_sampler = SubsetRandomSampler(temperature_train_indices)
temperature_test_sampler = SubsetRandomSampler(temperature_test_indices)
temperature_valid_sampler = SubsetRandomSampler(temperature_val_indices)

train_loader = DataLoader(
    dataset=temperature_dataset,
    batch_size = batch_size,
    sampler=temperature_train_sampler,
)

test_loader = DataLoader(
    dataset=temperature_dataset,
    batch_size = batch_size,
    sampler=temperature_test_sampler,
)

validation_loader = DataLoader(
    dataset=temperature_dataset,
    batch_size = batch_size,
    sampler=temperature_valid_sampler,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(6210,3105)
        self.fc2 = nn.Linear(3105,1035)
        self.fc3 = nn.Linear(1035,3105)
        self.fc4 = nn.Linear(3105,6210)
        '''
        self.fc1 = nn.Linear(11934,1326)
        self.fc2 = nn.Linear(1326,500)
        self.fc3 = nn.Linear(500,1326)
        self.fc4 = nn.Linear(1326,11934)
        '''
        self.act = nn.Tanh()
    
    def forward(self, x):
        
        output_batch_size = x.shape[0]
        x = x.view(-1, 6210)
        
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        
        out = out.view(output_batch_size,6,23,45)
        
        return out

# Test Function
def test(model, encoder, decoder, test_loader, device):

    # Load the model from the input model_path  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    correct = 0
    total = 0
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    best_worst_error_list = [1000000, 0]
    best_worst_output_list = [0, 0]
    best_worst_predicted_list = [0, 0]
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.view(inputs.shape[0], 1, 201, 401)
            labels = labels.view(labels.shape[0], 1, 201, 401)
            
            # Get output features and calculate loss
            inputs_encoded = encoder(inputs)
            labels_encoded = encoder(labels)
            
            outputs = model(inputs_encoded)
            loss = criterion(outputs, labels_encoded)
                
            # If the loss value is less than 0.01, we consider it being accurate
            for j in range(len(labels)):
                single_loss = criterion(outputs[j], labels_encoded[j])
                if single_loss.item() <= 0.01:
                    correct += 1
                total += 1
                
                # Record worst error
                if single_loss.item() > best_worst_error_list[1]:
                    best_worst_error_list[1] = single_loss.item()
       
                    best_worst_output_list[1] = labels[j]
                    best_worst_predicted_list[1] = outputs[j]
                    
                # Record best error
                if single_loss.item() < best_worst_error_list[0]:
                    best_worst_error_list[0] = single_loss.item()
                    
                    best_worst_output_list[0] = labels[j]
                    best_worst_predicted_list[0] = outputs[j]
                    

            # Add to the validation loss
            total_loss += loss.item()

    text_file = open('FNN_testingData_Gadi_1.txt', "w")
    n1 = text_file.write(str(total_loss)+"\n")
    n2 = text_file.write(str(100*correct//total)+"\n")
    text_file.close()
    print("Testing Data saved! The path is FNN_testingData_Gadi_1.txt")
    
    
    text_file = open('FNN_testingData_Gadi_2.txt', "w")
    n1 = text_file.write("/".join([str(elem) for elem in best_worst_error_list])+"\n")
    
    n2 = text_file.write("/".join([str("|".join([str(x) for x in elem.cpu().detach().numpy()[0].flatten()])) for elem in best_worst_output_list])+"\n")
    
    n3 = text_file.write("/".join([str("|".join([str(x) for x in decoder(encoder(elem.view(1, 1, 201, 401))).cpu().detach().numpy()[0][0].flatten()])) for elem in best_worst_output_list])+"\n")
    
    n4 = text_file.write("/".join([str("|".join([str(x) for x in decoder(elem.view(1, 6, 23, 45)).cpu().detach().numpy()[0][0].flatten()])) for elem in best_worst_predicted_list]))   
    
    text_file.close()
    print("Testing Data saved! The path is FNN_testingData_Gadi_2.txt")
    
model = Net().to(device)
test(model, encoder, decoder, test_loader, device)


# Further testing
best_evolution = [0,0,0,0]
worst_evolution = [0,0,0,0]
best_loss = 1e10
worst_loss = -1

# Test for a complete cycle of timestamps
for i in range(len(temperature_fields)):

    # Read the testing file
    testing_temperature_fields = temperature_fields[i]

    # Looping through the NN
    predicted_temperature_fields = [testing_temperature_fields[0]]
    single_predicted_temperature_fields = [testing_temperature_fields[0]]
    ConvAE_temperature_fields = [testing_temperature_fields[0]]

    testing_input = torch.from_numpy(testing_temperature_fields[0]).to(device)
    testing_latentSpace = encoder(testing_input.view(1, 1, 201, 401))
    
    original_predicted_distance = 0
    for i in range(99):
        # Output feed as input looping prediction
        testing_latentSpace = model(testing_latentSpace)
        predicted_temperature_fields.append(decoder(testing_latentSpace).cpu().detach().numpy()[0][0])
    
        # Single Prediction
        single_testing_input = torch.from_numpy(testing_temperature_fields[i]).to(device)
        single_testing_latentSpace = encoder(single_testing_input.view(1, 1, 201, 401))
        single_testing_latentSpace = model(single_testing_latentSpace)
        single_predicted_temperature_fields.append(decoder(single_testing_latentSpace).cpu().detach().numpy()[0][0])
    
        # Encoder-decoder transformation
        ConvAE_testing_input = torch.from_numpy(testing_temperature_fields[i+1]).to(device)
        ConvAE_temperature_fields.append(decoder(encoder(ConvAE_testing_input.view(1, 1, 201, 401))).cpu().detach().numpy()[0][0])
        
        original_predicted_distance += np.abs((decoder(testing_latentSpace).cpu().detach().numpy()[0][0] - testing_temperature_fields[i]).flatten()).sum()

    if original_predicted_distance < best_loss:
        best_loss = original_predicted_distance
        best_evolution[0] = testing_temperature_fields
        best_evolution[1] = predicted_temperature_fields
        best_evolution[2] = ConvAE_temperature_fields
        best_evolution[3] = single_predicted_temperature_fields
        
    
    if original_predicted_distance > worst_loss:
        worst_loss = original_predicted_distance
        worst_evolution[0] = testing_temperature_fields
        worst_evolution[1] = predicted_temperature_fields
        worst_evolution[2] = ConvAE_temperature_fields
        worst_evolution[3] = single_predicted_temperature_fields
    
best_evolution = np.asarray(best_evolution)
with open('FNN_best_evolution.npy', 'wb') as f:
    np.save(f, best_evolution)

worst_evolution = np.asarray(worst_evolution)
with open('FNN_worst_evolution.npy', 'wb') as f:
    np.save(f, worst_evolution)


# Testing for how many time steps we can use the trained FNN without loosing track of the transient dynamics
testing_time_steps = [1, 2, 4, 8, 16, 99]


PCA_lists = [[] for x in testing_time_steps]
PCA_relative_lists = [[] for x in testing_time_steps]
loss_lists = [[] for x in testing_time_steps]
best_time_step = -1

def SVD(X):
    U,Sigma,VT = np.linalg.svd(X,full_matrices=0)
    return U, Sigma, VT

for i in range(len(temperature_fields)):

    # Read the testing file
    testing_temperature_fields = temperature_fields[i]

    # Looping through the NN
    predicted_temperature_fields_list = [[testing_temperature_fields[0]] for x in testing_time_steps]
    testing_input_list = [encoder(torch.from_numpy(testing_temperature_fields[0]).to(device).view(1, 1, 201, 401)) for x in testing_time_steps]
    
    for i in range(99):
        for j in range(len(testing_time_steps)):
            if i % testing_time_steps[j] == 0:
                testing_input_list[j] = encoder(torch.from_numpy(testing_temperature_fields[i]).to(device).view(1, 1, 201, 401))
            testing_latentSpace_j = model(testing_input_list[j])
            testing_input_list[j] = testing_latentSpace_j
            predicted_temperature_fields_list[j].append(decoder(testing_latentSpace_j).cpu().detach().numpy()[0][0])

    # Calculate the PCA difference and data difference for each time step
    _, S_original, _ = SVD(np.transpose(np.asarray(testing_temperature_fields),(1,2,0)))
    for j in range(len(testing_time_steps)):
        _, S_predicted_j, _ = SVD(np.transpose(np.asarray(predicted_temperature_fields_list[j]),(1,2,0)))
        PCA_lists[j].append(np.linalg.norm(S_predicted_j.diagonal() - S_original.diagonal()))
        PCA_relative_lists[j].append(np.linalg.norm(S_predicted_j.diagonal() - S_original.diagonal()) / np.linalg.norm(S_original.diagonal()))
        loss_lists[j].append(np.linalg.norm(predicted_temperature_fields_list[j] - testing_temperature_fields))


PCA_lists = np.asarray(PCA_lists)
with open('FNN_testing_PCA_lists.npy', 'wb') as f:
    np.save(f, PCA_lists)

PCA_relative_lists = np.asarray(PCA_relative_lists)
with open('FNN_testing_PCA_relative_lists.npy', 'wb') as f:
    np.save(f, PCA_relative_lists)

loss_lists = np.asarray(loss_lists)
with open('FNN_testing_loss_lists.npy', 'wb') as f:
    np.save(f, loss_lists)



# Visualize further testing 

# Read the testing file
testing_temperature_fields = temperature_fields[123]

# Looping through the NN
predicted_temperature_fields_list = [[testing_temperature_fields[0]] for x in testing_time_steps]
testing_input_list = [encoder(torch.from_numpy(testing_temperature_fields[0]).to(device).view(1, 1, 201, 401)) for x in testing_time_steps]
    
for i in range(99):
    for j in range(len(testing_time_steps)):
        if i % testing_time_steps[j] == 0:
            testing_input_list[j] = encoder(torch.from_numpy(testing_temperature_fields[i]).to(device).view(1, 1, 201, 401))
        testing_latentSpace_j = model(testing_input_list[j])
        testing_input_list[j] = testing_latentSpace_j
        predicted_temperature_fields_list[j].append(decoder(testing_latentSpace_j).cpu().detach().numpy()[0][0])

predicted_temperature_fields_list = np.asarray(predicted_temperature_fields_list)
with open('FNN_further_testing.npy', 'wb') as f:
    np.save(f, predicted_temperature_fields_list)

testing_temperature_fields = np.asarray(testing_temperature_fields)
with open('FNN_further_testing_reference.npy', 'wb') as f:
    np.save(f, testing_temperature_fields)