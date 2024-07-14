# Loading packages
import os
import pprint
import sys
import warnings
from typing import List

# Ignore future warnings that arises from BertModel
warnings.filterwarnings("ignore", category=FutureWarning)

# Importing base modules
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from transformers import BertModel

import wandb

# Inserting path to AmazonData class object
sys.path.insert(0, os.getcwd() + "/src/data/")
from AmazonData import AmazonData

# Setting device to cpu or cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')


# Defining sweep
sweep_config = {"method": "random"}

# Setting parameters
parameters_dict = {
    "optimizer": {"values": ["sgd", "adam"]},
    "batch_size": {"values": [100,150,200,250]},
    "drop_out": {"values": [0.15,0.25,0.35]},
    "lr": {"values": [0.01,0.001]},
}
sweep_config["parameters"] = parameters_dict

# Print the defined parameters
pprint.pprint(sweep_config)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="Amazon-Reviews-hpc", entity="amazonproject")

# Define loss function
loss_fn = nn.CrossEntropyLoss().to(device)

# Define model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int, p: float):
        super(SentimentClassifier, self).__init__()
        # load pretrained BERT-model
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.drop = nn.Dropout(p=p)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

# Build data loader function
def build_dataLoader(data, batch_size: int):
    return DataLoader(data, batch_size=batch_size, shuffle=True)

# Build optimizer
def build_optimizer(opt, model, lr: float):
    if opt == 'adam':
        return optim.Adam(model.parameters(), lr=lr, betas=(0.85,0.89), weight_decay=1e-3)
    else:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

# define train loop
def train_epoch(model, trainloader, optimizer) -> List:
    running_loss = 0
    running_acc = 0
    num_batches = len(trainloader)
    for batch_idx, data in enumerate(trainloader):
        print(f"Batch {batch_idx+1} of {num_batches}")
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["targets"].to(device)
        # zero the gradients
        optimizer.zero_grad()
        # Make predictions
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # Calculate loss
        loss = loss_fn(output, labels)
        running_loss += loss.item()
        # Backpropagate
        loss.backward()
        # Take a step
        optimizer.step()
        # Softmax the predictions and get the indices
        y_pred = F.softmax(output, dim=1).argmax(dim=1)
        running_acc += ((y_pred == labels).sum()/ labels.shape[0]).item() / num_batches
        # Log batch result
        wandb.log(
            {
                " (Batch loss)": running_loss,
                " (Batch accuracy)": running_acc,
            }
        )
        if ((batch_idx + 1) % 2) == 0:
            print(
                f"Loss: {running_loss} \tAccuracy: {round(running_acc,4) * 100}%"
            )
    return running_loss / num_batches, running_acc

# Build model
def build_model(dropout: float):
    return SentimentClassifier(n_classes=3, p=dropout)

# Train model
def train(config=None):
    """
    Function to train model and store training results
    Requirements:
        - Data must have been generated before executing this script

    Parameters:
        (OPTIONAL)
        --lr: learning rate
        --epochs: Number of training loops
        --batch_size: Batch size

    Outputs:
        - models/final_model.pth

    """
    # init wandb
    with wandb.init(config=config):
        # extract configurations
        config = wandb.config
        print("Initializing training")
        # build model
        model = build_model(config.drop_out).to(device)
        # load trainset
        train_set = torch.load('data/processed/train.pth')
        # create Dataloader
        trainloader = build_dataLoader(train_set, config.batch_size)
        # build optimizer
        optimizer = build_optimizer(config.optimizer, model, config.lr)
        # number of epochs
        num_epochs = 10
        # Set the model to train mode
        model.train()
        for i in range(num_epochs):
            print(f"Epoch {i+1} of {num_epochs}")
            # train epoch
            avg_loss, avg_acc = train_epoch(model, trainloader, optimizer)
            print(
                f"Epoch {i+1} loss: {avg_loss} \tEpoch acc: {avg_acc}"
            )
            # log epoch results
            wandb.log(
                {
                    "Loss": avg_loss,
                    "Accuracy": avg_acc
                }
            )
        # save model
        torch.save(
            model,
            f"models/model_opt{config.optimizer}_bs{config.batch_size}_do{config.drop_out}_lr{config.lr}.pth",
        )

# python
if __name__ == "__main__":
    # run the agent
    wandb.agent(sweep_id, train, count=15)
