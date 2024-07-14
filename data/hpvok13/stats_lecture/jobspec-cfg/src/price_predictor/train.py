import os
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from model import TransformerDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AttrDict

from data import StockDataset


def train_epoch(
    config: AttrDict,
    model: TransformerDecoder,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
) -> float:
    model.train()
    train_loss = 0.0
    running_loss = 0.0
    running_correct = 0

    bar = tqdm(
        train_loader,
        desc=(f"Training | Epoch: {epoch} | " f"Loss: {0:.4f}"),
    )
    for i, seq in enumerate(bar):
        seq = seq.to(device)

        # Forward pass
        outputs = model(seq)
        loss = criterion(outputs, seq)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Update statistics
        train_loss += loss.item()
        running_loss += loss.item()

        # Log statistics
        if (i + 1) % config.train.log_freq == 0:
            running_loss /= config.train.log_freq
            wandb_log(
                dict(
                    running_loss=running_loss,
                )
            )
            bar.set_description(
                f"Training | Epoch: {epoch} | " f"Loss: {running_loss:.4f}"
            )
            running_loss = 0

    # Calculate average training loss and accuracy
    train_loss /= len(train_loader)
    wandb_log(dict(train_loss=train_loss))

    return train_loss


def eval_epoch(
    model: TransformerDecoder,
    criterion: torch.nn.Module,
    test_loader: DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
) -> float:
    model.eval()
    test_loss = 0.0
    bar = tqdm(
        test_loader,
        desc=(f"Testing | Epoch: {epoch}"),
    )

    with torch.no_grad():
        for seq in bar:
            seq = seq.to(device)

            # Forward pass
            outputs = model(seq)
            loss = criterion(outputs, seq)

            # Update statistics
            test_loss += loss.item()

    # Calculate average test loss and accuracy
    test_loss /= len(test_loader)

    wandb_log(dict(test_loss=test_loss, epoch=epoch))

    return test_loss


def train(config: AttrDict) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (AttrDict): Configuration parameters.
    """
    torch.set_float32_matmul_precision(config.train.matmul_precision)
    # Get device and initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDecoder(**config.model).to(device)

    # Compile the model if requested
    model = torch.compile(
        model,
        fullgraph=config.compile.fullgraph,
        dynamic=config.compile.dynamic,
        backend=config.compile.backend,
        mode=config.compile.mode,
        disable=config.compile.disable,
    )

    # Initialize the optimizer
    if config.optimizer.fn == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    elif config.optimizer.fn == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optimizer.fn} not implemented"
        )

    # Initialize the loss function
    if config.criterion == "mse_loss":
        criterion = torch.nn.MSELoss()
    elif config.criterion == "rmse_loss":
        mse_loss = torch.nn.MSELoss()
        criterion = lambda x: torch.sqrt(mse_loss(x))
    else:
        raise NotImplementedError(
            f"Criterion {config.criterion} not implemented"
        )

    # Get the data loaders
    train_dataset = StockDataset(**config.dataset, train=True)
    val_dataset = StockDataset(**config.dataset, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    test_loader = DataLoader(val_dataset, batch_size=config.train.batch_size)

    # Initialize Weights & Biases
    if config.wandb:
        wandb.init(project="Price Predictor", config=config)
        wandb_log = lambda x: wandb.log(x)
    else:
        wandb_log = lambda x: None

    for epoch in range(config.train.epochs):
        # Train the model
        train_loss = train_epoch(
            config,
            model,
            optimizer,
            criterion,
            train_loader,
            wandb_log,
            epoch,
            device,
        )

        # # Evaluate the model on the test set
        test_loss = eval_epoch(
            model, criterion, test_loader, wandb_log, epoch, device
        )

        # Print the epoch statistics
        print(
            f"Epoch [{epoch}/{config.train.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f}, "
        )

        # Save Model
        file_path = os.path.abspath(
            os.path.join(config.train.model_dir, f"model_{epoch}.pt")
        )
        link_path = os.path.abspath(
            os.path.join(config.train.model_dir, "model.pt")
        )
        torch.save(model, file_path)
        try:
            os.remove(link_path)
        except FileNotFoundError:
            pass
        os.symlink(file_path, link_path)


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)

    train(config)
