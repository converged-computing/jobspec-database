import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
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
model_path = "lstm_best.pth"

#Parameters
batch_size = 1

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

'''
# Temperature for the two consecutive timestamp
temperature_fields = []

# Folder Path
path = "/scratch/kr97/xh7958/comp4560/solutions"
  
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
'''
temperature_fields = np.load('/scratch/kr97/xh7958/comp4560/solutions_standard.npy')


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
    
    def __init__(self, dataset):
        self.input = dataset[:,:50,:,:]
        self.output = dataset[:,50:,:,:]
        
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_item = self.input[index]
        output_item = self.output[index]
        
        return input_item, output_item


temperature_dataset = KMNIST(
    temperature_fields
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
        self.lstm1 = nn.LSTM(input_size=6210, hidden_size=3105, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=3105, hidden_size=6210, num_layers=1, batch_first=True)
        
    
    def forward(self, x):
        sequence_length = 50
        
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        
        out = out.view(sequence_length,6,23,45)
        
        return out
    
    
def test(model, encoder, decoder, test_loader, device):

    # Load the model from the input model_path  
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    criterion = nn.MSELoss()
    total_loss = 0.0
    
    best_worst_error_list = [1000000, 0]
    best_worst_output_list = [0, 0]
    best_worst_predicted_list = [0, 0]
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            encoded_inputs = encoder(inputs.view(inputs.shape[1], 1, 201, 401)).reshape(batch_size, 50, -1)
            encoded_labels = encoder(labels.view(labels.shape[1], 1, 201, 401))

            # Get output features, calculate loss and optimize
            outputs = model(encoded_inputs)
            loss = criterion(outputs.float(), encoded_labels.float())

            for j in range(len(encoded_labels)):
                single_loss = criterion(outputs[j], encoded_labels[j])
                # Record worst error
                if single_loss.item() > best_worst_error_list[1]:
                    best_worst_error_list[1] = single_loss.item()
                    best_worst_output_list[1] = labels[0][j]
                    best_worst_predicted_list[1] = outputs[j]
                    
                # Record best error
                if single_loss.item() < best_worst_error_list[0]:
                    best_worst_error_list[0] = single_loss.item()
                    best_worst_output_list[0] = labels[0][j]
                    best_worst_predicted_list[0] = outputs[j]
                    

            # Add to the validation loss
            total_loss += loss.item()

    text_file = open('LSTM_testingData_Gadi_1.txt', "w")
    n1 = text_file.write(str(total_loss)+"\n")
    text_file.close()
    print("Testing Data saved! The path is LSTM_testingData_Gadi_1.txt")
    
    
    text_file = open('LSTM_testingData_Gadi_2.txt', "w")
    n1 = text_file.write("/".join([str(elem) for elem in best_worst_error_list])+"\n")
    
    n2 = text_file.write("/".join([str("|".join([str(x) for x in elem.cpu().detach().numpy().flatten()])) for elem in best_worst_output_list])+"\n")
    
    
    n3 = text_file.write("/".join([str("|".join([str(x) for x in decoder(encoder(elem.view(1, 1, 201, 401)).view(1, 6, 23, 45)).cpu().detach().numpy()[0][0].flatten()])) for elem in best_worst_output_list])+"\n")
    
    n4 = text_file.write("/".join([str("|".join([str(x) for x in decoder(elem.view(1, 6, 23, 45)).cpu().detach().numpy()[0][0].flatten()])) for elem in best_worst_predicted_list]))   
    
    text_file.close()
    print("Testing Data saved! The path is LSTM_testingData_Gadi_2.txt")
    
    
    
model = Net().to(device)
test(model, encoder, decoder, test_loader, device)


# Further testing

best_evolution = [0,0,0]
worst_evolution = [0,0,0]
best_loss = 1e10
worst_loss = -1

# Test for a complete cycle of timestamps
for i in range(len(temperature_fields)):

    # Read the testing file
    testing_temperature_fields = temperature_fields[i]

    # Calculate predicted temperature fields and compressed-decompressed field
    temperature_fields_inputs = testing_temperature_fields[:50,:,:]
    temperature_fields_outputs = testing_temperature_fields[50:,:,:]
    
    encoded_inputs = encoder(torch.from_numpy(temperature_fields_inputs).to(device).view(50, 1, 201, 401)).reshape(1, 50, -1)
    predicted_outputs = model(encoded_inputs)
    
    decoded_predicted = decoder(predicted_outputs.view(50, 6, 23, 45)).cpu().detach().numpy().reshape(50,201,401)
    decoded_original = decoder(encoded_inputs.view(50, 6, 23, 45)).cpu().detach().numpy().reshape(50,201,401)
    
    original_predicted_distance = np.abs(temperature_fields_outputs - decoded_predicted).flatten().sum()
    
    if original_predicted_distance < best_loss:
        best_loss = original_predicted_distance
        best_evolution[0] = temperature_fields_outputs
        best_evolution[1] = decoded_predicted
        best_evolution[2] = decoded_original
    
    if original_predicted_distance > worst_loss:
        worst_loss = original_predicted_distance
        worst_evolution[0] = temperature_fields_outputs
        worst_evolution[1] = decoded_predicted
        worst_evolution[2] = decoded_original

best_evolution = np.asarray(best_evolution)
with open('LSTM_best_evolution.npy', 'wb') as f:
    np.save(f, best_evolution)

worst_evolution = np.asarray(worst_evolution)
with open('LSTM_worst_evolution.npy', 'wb') as f:
    np.save(f, worst_evolution)
        
    