from algorithms.agent import WorldModel, ImagBehavior
from ruamel.yaml import YAML
from data.pipeline import Pipeline
import torch
from tqdm import tqdm
from modules import tools
import numpy as np
import csv
import os

data_directory = "data/SimulatedData4-13/mixed/test"
model_path = "models/SimulatedDataModel4-16/checkpoint.pt"

def main():
    with open("configs.yaml") as f:
        yaml = YAML()
        configs = yaml.load(f)

    obs_space = 11
    act_space = 4
    batch_size = 256

    # Open the CSV file, convert it to a list of rows, and count the number of rows
    with open(os.path.join(data_directory, 'states.csv'), 'r') as file:
        reader = csv.reader(file)
        sequence_length = len(list(reader)) - 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = WorldModel(obs_space, act_space, configs)

    model.requires_grad_(requires_grad=False)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    model.to(configs['device'])

    pipeline = Pipeline(os.path.join(data_directory, "states.csv"), os.path.join(data_directory, "actions.csv"), os.path.join(data_directory, "rewards.csv"))
    dataloader = pipeline.read_csv(sequence_length=sequence_length, batch_size=batch_size)

    history_size = configs['rssm']['history_size']

    _, (states, actions, rewards) = next(enumerate(iter(dataloader)))

    states = states.to(configs['device'])
    actions = actions.to(configs['device'])
    rewards = rewards.to(configs['device'])

    outputs = model._valid({'state': states, 'action': actions, 'reward': rewards})

    outputs = outputs.to('cpu')  # Move the tensor to CPU memory

    outputs_numpy = outputs.squeeze(0).detach().numpy()
    
    np.savetxt('data/testSequenceOutput4-14.csv', outputs_numpy, delimiter=',', fmt='%.8f')

if __name__ == "__main__":
    main()