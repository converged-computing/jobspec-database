"""
Project: Instrument Classification
File: run_model.py
Authors: John Zontos
Date: June 5, 2023

Description:
Demo script.

Author: John Zontos

This script serves as the main entry point for the Conditional Musical Instrument Classification System.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# from modules.audio_transformer import AudioTransformer
from modules.audio_lstm import AudioLSTM
# from modules.audio_transformer import AudioTransformer
from modules.data import AudioClassificationDataset
from modules.audio_nn import AudioNN
from modules.trainer import Trainer
from utils.metrics import calculate_metrics, print_results

# from sklearn.model_selection import train_test_split
import argparse
import random
import tqdm as tq
import sys
import os

import neptune
from neptune.utils import stringify_unsupported

print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(root_dir, csv_file, params):
    # Instantiate Neptune
    run = neptune.init_run(
        project="Soundbendor/instrument-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZjY2NzQ5Yi0wZDVjLTRiODktOTNlNy0xODg3YTRkZTVmYTcifQ==",
    )

    # neptune tags
    run['sys/tags'].add(["LSTM", "test"])

    # Instantiate the dataset
    wav_length = 16000
    num_samples = 20000
    dataset = AudioClassificationDataset(root_dir, csv_file, wav_length, num_samples=num_samples)
    print("INFO: Dataset loaded.")
    dataset.check_audio()
    dataset.info()
    # TODO: add logging instead of printing.

    params.update({
        "input_size": dataset.wav_length,
        "num_classes": dataset.num_classes,
        "num_samples": len(dataset) if num_samples is None else num_samples
        })
    
    run["logging/hyperparams"] = stringify_unsupported(params)

    # Set a random seed for reproducibility and shuffle dataset
    random.seed(42)

    # Set the proportions for train, validation, and test splits
    train_split = 1 - (params["test_split"] + params["val_split"])

    # Perform the splits
    train_data, val_data, test_data = random_split(dataset, [train_split, params["val_split"], params["test_split"]])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params["batch_size"])
    test_loader = DataLoader(test_data, batch_size=params["batch_size"])

    print("INFO: Dataloaders created.")

    # Instantiate the model
    model = AudioLSTM(params["input_size"], params["hidden_dim"], params["num_layers"], params["num_classes"], params["bidirectional"], params["dropout_rate"])

    # n_mfcc=40
    # model = AudioNN(input_size=n_mfcc, num_classes=params["num_classes"])
    print("INFO: Model created.")

    if torch.cuda.device_count() > 1:
        print("Multiple GPUs available, using: " + str(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Create model trainer
    trainer = Trainer(model, criterion, optimizer, run, device)
    print("INFO: Trainer Created.")

    # Training loop
    total_steps = len(train_loader)
    best_val_accuracy = None
    for epoch in range(params["num_epochs"]):
        # training phase
        pbar = tq.tqdm(desc="Epoch {}".format(epoch+1), total=total_steps, unit="steps")
        avg_train_loss, train_accuracy = trainer.train(train_loader, pbar)
        pbar.close()
            
        # Validation phase
        avg_val_loss, val_accuracy = trainer.validate(val_loader)

        if not best_val_accuracy or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # torch.save(model, 'audio_lstm_model.pt')

        # Print progress
        print('------------------ Epoch [{}/{}] ------------------'.format(epoch + 1, params["num_epochs"]))
        print('Training Loss: {:.4f} Training Accuracy: {:.2f}'.format(avg_train_loss, train_accuracy))
        print('Validation Loss: {:.4f} Validation Accuracy: {:.2f}'.format(avg_val_loss, val_accuracy))

        run["logging/train/epoch/loss"].append(avg_train_loss)
        run["logging/train/epoch/acc"].append(train_accuracy)

        run["logging/validation/epoch/loss"].append(avg_val_loss)
        run["logging/validation/epoch/acc"].append(val_accuracy)

        
    # Evaluation
    print("INFO: Evaluation.")
    predicted, scores, y_true = trainer.evaluate(test_loader)
    results = calculate_metrics(predicted, scores, y_true, dataset.num_classes)
    print_results(results, best_val_accuracy)
    run['results'] = results
    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with hyperparameters')

    # Add data directory and hyperparameters as arguments
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--num_layers', type=int)

    args = parser.parse_args()

    root_dir = args.data_dir
    csv_file = os.path.join(root_dir, "dataset.csv")

    params = {
        "model_name": "audio_lstm",
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout_rate,
        'num_layers': args.num_layers,
        "num_epochs": 25,
        "val_split": 0.1,
        "test_split": 0.1,
        "device": device,
        "bidirectional": True,
    }

    # Call the main function with the provided paths
    main(root_dir, csv_file, params)