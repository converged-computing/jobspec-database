import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from models import SRResNet
from dataset import SRDataset
# Utility functions
from utils import parse_arguments, read_settings, save_checkpoint, clip_gradient
from logger import Logger
from tqdm import tqdm  # For displaying progress during training loops
import numpy as np
from evaluate import evaluate  # Evaluation script to measure performance metrics

# Set device based on available hardware
device = torch.device("cuda" if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else "cpu")

# Enable faster convolution operations when the input size does not vary
cudnn.benchmark = True


def main():
    """
    Main function to handle setup, training, and evaluation of the super-resolution model.
    """
    print(f'{device = }')

    # Parse command-line arguments or configuration files
    args = parse_arguments()

    # Read training, generator, and dataset settings from configuration file specified by the arguments
    settings = read_settings(args.config)
    generator_settings = settings.get('generator', {})
    train_settings = settings.get('train', {})
    dataset_settings = settings.get('dataset', {})
    train_dataloader_settings = settings.get('train_dataloader', {})
    test_dataloader_settings = settings.get('test_dataloader', {})

    # Print out settings for verification
    print(f'{generator_settings = }\n{train_settings = }\n{dataset_settings = }\n'
          f'{train_dataloader_settings = }\n{test_dataloader_settings = }')

    # Initialize datasets and data loaders
    train_dataset = SRDataset(**dataset_settings, stage='train',
                              lr_img_type='imagenet-norm', hr_img_type='[-1, 1]')
    train_loader = DataLoader(
        train_dataset, **train_dataloader_settings, pin_memory=True)

    test_dataset = SRDataset(**dataset_settings, stage='test',
                             lr_img_type='imagenet-norm', hr_img_type='[-1, 1]')
    test_loader = DataLoader(
        test_dataset, **test_dataloader_settings, pin_memory=True)

    # Initialize the super-resolution model
    model = SRResNet(**generator_settings,
                     scaling_factor=dataset_settings['scaling_factor'])

    # Setup logging
    logger = Logger(settings, str(model), 'INM705-SuperResolution')

    # Start the training process
    train(model, train_loader, test_loader, logger, **train_settings)


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, logger: Logger, lr: float, epochs: int,
          grad_clip=None):
    """
    Train the model across multiple epochs, evaluating and saving checkpoints periodically.

    :param model: The neural network model to train.
    :param train_loader: DataLoader for the training data.
    :param test_loader: DataLoader for the testing data.
    :param logger: Logger object to record training progress.
    :param lr: Learning rate for the optimizer.
    :param epochs: Total number of epochs to train.
    :param grad_clip: Maximum norm of the gradients for clipping.
    """
    # Initialize the optimizer
    optimizer = torch.optim.Adam(params=filter(
        lambda p: p.requires_grad, model.parameters()), lr=float(lr))
    model = model.to(device)
    criterion = nn.MSELoss().to(device)  # Define the loss function

    # Loop over each epoch
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        epoch_loss = train_epoch(
            model, train_loader, criterion, optimizer, grad_clip)
        train_end = time.perf_counter()

        # Evaluate model performance on the test dataset
        psnrs, ssims, wandb_images = evaluate(model, test_loader)
        eval_end = time.perf_counter()

        # Log training and evaluation results
        logger.log({
            'epoch_loss': epoch_loss,
            'epoch_train_time': train_end - epoch_start,
            'epoch_eval_time': eval_end - train_end,
            'mean_psnr': np.mean(psnrs),
            'mean_ssim': np.mean(ssims),
            **wandb_images
        })

        # Print out loss and time taken for each epoch
        print(f'{epoch = }\n{epoch_loss = }\n'
              f'epoch_train_time: {train_end - epoch_start}\n'
              f'epoch_eval_time: {eval_end - train_end}\n'
              f'mean_psnr: {np.mean(psnrs)}\n'
              f'mean_ssim: {np.mean(ssims)}\n')

        # Save model state at the end of the epoch
        save_checkpoint(epoch, model, str(model), optimizer)


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, grad_clip=None):
    """
    Handle the training of one epoch, updating model weights per batch.

    :param model: The super-resolution model being trained.
    :param train_loader: DataLoader for the training dataset.
    :param criterion: Loss function to measure error.
    :param optimizer: Optimizer to adjust model weights.
    :param grad_clip: Threshold for gradient clipping to avoid exploding gradients.
    """
    model.train()  # Enable training mode for batch normalization and dropout

    total_loss = 0  # Track loss across all batches

    # Process each batch from the data loader
    for lr_imgs, hr_imgs in tqdm(train_loader):
        # Transfer low-res images to the device (GPU/CPU)
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)  # Transfer high-res images to the device

        # Forward pass to compute super-resolved images
        sr_imgs = model(lr_imgs)

        # Compute loss between super-resolved images and high-res targets
        loss = criterion(sr_imgs, hr_imgs)

        # Backward pass to compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping
        if grad_clip is not None:
            clip_gradient(optimizer, float(grad_clip))

        # Update model weights
        optimizer.step()

        total_loss += loss.item()  # Accumulate the loss

    del lr_imgs, hr_imgs, sr_imgs  # Free up memory

    return total_loss / len(train_loader)  # Return average loss


if __name__ == '__main__':
    main()
