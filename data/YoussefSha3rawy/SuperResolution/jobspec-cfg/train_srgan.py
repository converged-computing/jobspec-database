import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from models import SRResNet, TruncatedVGG19, Discriminator
from dataset import SRDataset
from utils import parse_arguments, read_settings, save_checkpoint, adjust_learning_rate, clip_gradient, calculate_gradient_penalty, convert_image
from logger import Logger
import numpy as np
from tqdm import tqdm
from torchvision.models import efficientnet_b0
from evaluate import evaluate
from torch.utils.data import DataLoader

# Set the preferred device to GPU if available, otherwise to MPS (Apple's Metal Performance Shaders) or CPU as fallback
device = torch.device("cuda" if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else "cpu")

# Optimizes performance by allowing cudnn to use non-deterministic algorithms
cudnn.benchmark = True


def main():
    """
    Main function to setup and initiate the training and evaluation processes.
    """
    print(f'Device in use: {device}')

    # Parse command line arguments for configuration paths or other options
    args = parse_arguments()

    # Load settings from a YAML configuration file
    settings = read_settings(args.config)

    # Extract settings for different components
    generator_settings = settings.get('generator', {})
    discriminator_settings = settings.get('discriminator', {})
    train_settings = settings.get('train', {})
    dataset_settings = settings.get('dataset', {})
    train_dataloader_settings = settings.get('train_dataloader', {})
    test_dataloader_settings = settings.get('test_dataloader', {})

    # Print out loaded settings for verification
    print(f'Generator Settings: {generator_settings}\nDiscriminator Settings: {discriminator_settings}\nTraining Settings: {train_settings}\nDataset Settings: {dataset_settings}\nTrain DataLoader Settings: {train_dataloader_settings}\nTest DataLoader Settings: {test_dataloader_settings}')

    # Initialize the generator, potentially with settings for scaling factor
    generator = SRResNet(**generator_settings,
                         scaling_factor=dataset_settings['scaling_factor'])

    # Initialize the discriminator based on type specified in settings; default to a custom Discriminator if not using EfficientNet
    if discriminator_settings['discriminator_type'] == 'EfficientNet':
        discriminator = efficientnet_b0(pretrained=False)
        discriminator.classifier[1] = nn.Linear(
            discriminator.classifier[1].in_features, 1)
    else:
        del discriminator_settings['discriminator_type']
        discriminator = Discriminator(**discriminator_settings)

    # Initialize a truncated VGG19 model used for content loss computation
    truncated_vgg19 = TruncatedVGG19(i=36)
    truncated_vgg19.eval()

    # Move all models to the default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)

    # Setup dataloaders for training and testing datasets
    train_dataset = SRDataset(**dataset_settings, stage='train',
                              lr_img_type='imagenet-norm', hr_img_type='imagenet-norm')
    train_loader = DataLoader(
        train_dataset, **train_dataloader_settings, pin_memory=True)
    test_dataset = SRDataset(**dataset_settings, stage='test',
                             lr_img_type='imagenet-norm', hr_img_type='[-1, 1]')
    test_loader = DataLoader(
        test_dataset, **test_dataloader_settings, pin_memory=True)

    # Initialize logging utility
    logger = Logger(settings, discriminator.__class__.__name__,
                    'INM705-SuperResolution')

    # Start the training process
    train(generator, discriminator, truncated_vgg19,
          train_loader, test_loader, logger, **train_settings)


def train(generator, discriminator, truncated_vgg19, train_loader, test_loader, logger, epochs, lr_g, lr_d, loss_type='Default', **kwargs):
    """
    Train the super-resolution model using GAN architecture.

    Parameters:
    - generator: the generator model from GAN
    - discriminator: the discriminator model from GAN
    - truncated_vgg19: truncated VGG19 model for perceptual loss calculation
    - train_loader: DataLoader for the training dataset
    - test_loader: DataLoader for the testing dataset
    - logger: utility for logging training progress and results
    - epochs: number of epochs to train
    - lr_g: learning rate for the generator
    - lr_d: learning rate for the discriminator
    - loss_type: specifies the type of loss function to use
    - kwargs: additional parameters for future use
    """
    # Convert learning rates to floats (necessary if values are passed as strings from command line)
    lr_g = float(lr_g)
    lr_d = float(lr_d)

    # Initialize optimizers for both generator and discriminator
    optimizer_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()), lr=lr_g)
    optimizer_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr_d)

    # Set up loss functions
    content_loss_criterion = nn.MSELoss().to(device)
    adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    # Start training loop over specified epochs
    for epoch in range(1, epochs + 1):
        # Adjust learning rates mid-training if specified
        if epoch == int(epochs // 2):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # Start timing this epoch
        epoch_start = time.perf_counter()

        # Train for one epoch
        epoch_loss = train_epoch(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion, optimizer_g, optimizer_d, beta=kwargs.get(
            'beta', 1), alpha=kwargs.get('alpha', 1), loss_type=loss_type, grad_clip=kwargs.get('grad_clip'), lambda_gp=kwargs.get('lambda_gp', 10))

        # End timing for training
        train_end = time.perf_counter()

        # Evaluate the model
        psnrs, ssims, images_dict = evaluate(generator, test_loader)
        eval_end = time.perf_counter()

        # Log training and evaluation results
        logger.log({
            'epoch_train_time': train_end - epoch_start,
            'epoch_eval_time': eval_end - train_end,
            'mean_psnr': np.mean(psnrs),
            'mean_ssim': np.mean(ssims),
            **epoch_loss,
            **images_dict
        })

        # Print epoch results
        print({
            'epoch': epoch,
            'epoch_train_time': train_end - epoch_start,
            'epoch_eval_time': eval_end - train_end,
            'mean_psnr': np.mean(psnrs),
            'mean_ssim': np.mean(ssims),
            **epoch_loss
        })

        # Save state for both generator and discriminator
        save_checkpoint(
            epoch, generator, f'{str(generator)}_{discriminator.__class__.__name__}{f"_{loss_type}" if loss_type != "Default" else ""}', optimizer_g)
        save_checkpoint(epoch, discriminator,
                        f'{discriminator.__class__.__name__}{f"_{loss_type}" if loss_type != "Default" else ""}', optimizer_d)


def train_epoch(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
                optimizer_g, optimizer_d, beta, alpha, loss_type, grad_clip=None, lambda_gp=10):
    """
    Executes training operations for a single epoch.

    Parameters:
    - train_loader: DataLoader providing batches of training data.
    - generator: GAN generator model, creates super-resolved images from low-resolution inputs.
    - discriminator: GAN discriminator model, differentiates between real (high-resolution) and generated (super-resolved) images.
    - truncated_vgg19: Truncated VGG19 network for calculating perceptual (content) loss using feature maps.
    - content_loss_criterion: MSE loss function for measuring content loss.
    - adversarial_loss_criterion: Binary cross-entropy loss for measuring adversarial loss.
    - optimizer_g: Optimizer for updating the generator's weights.
    - optimizer_d: Optimizer for updating the discriminator's weights.
    - beta: Weighting factor for adversarial loss in the total loss calculation.
    - alpha: Weighting factor for mean squared error in the total perceptual loss.
    - loss_type: Type of GAN loss to use, supports 'Default' or 'WGAN'.
    - grad_clip: Optional gradient clipping to prevent exploding gradients.
    - lambda_gp: Lambda for gradient penalty, applicable in WGAN-GP loss calculation.

    Returns:
    - Dictionary containing the average losses computed over the epoch.
    """
    beta = float(beta)
    alpha = float(alpha)
    # Set both models to training mode
    generator.train()
    discriminator.train()

    # Initialize loss accumulators
    total_content_loss = 0
    total_mse_loss = 0
    total_adversarial_loss = 0
    total_perceptual_loss = 0
    total_discriminator_loss = 0

    # Iterate over batches from the DataLoader
    for lr_imgs, hr_imgs in tqdm(train_loader):
        # Move low-resolution images to the computation device
        lr_imgs = lr_imgs.to(device)
        # Move high-resolution images to the computation device
        hr_imgs = hr_imgs.to(device)

        # GENERATOR UPDATE STEP
        # Generate super-resolved images from low-res inputs
        sr_imgs = generator(lr_imgs)

        # Convert the generated images to the format expected by the VGG19 network
        sr_imgs = convert_image(
            sr_imgs, source='[-1, 1]', target='imagenet-norm')

        # Calculate feature maps in VGG space for SR and HR images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(
            hr_imgs).detach()  # Detach to stop gradients

        # Discriminate super-resolved images to calculate adversarial loss
        sr_discriminated = discriminator(sr_imgs)

        # Compute content loss and adversarial loss
        content_loss = content_loss_criterion(
            sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        mse_loss = content_loss_criterion(sr_imgs, hr_imgs)
        if loss_type == 'WGAN':
            # For WGAN, loss is the negative mean of discriminator outputs
            adversarial_loss = -sr_discriminated.mean()
        else:
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated))

        # Calculate total perceptual loss as a weighted sum of content, adversarial, and MSE losses
        perceptual_loss = content_loss + beta * adversarial_loss + alpha * mse_loss

        # Backpropagate generator loss
        optimizer_g.zero_grad()
        perceptual_loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)  # Optionally clip gradients
        optimizer_g.step()

        # DISCRIMINATOR UPDATE STEP
        # Discriminate both high-resolution (real) and super-resolved (fake) images
        hr_discriminated = discriminator(hr_imgs)
        # Detach SR images to avoid backpropagating through generator
        sr_discriminated = discriminator(sr_imgs.detach())

        # Calculate discriminator loss using Binary Cross-Entropy or WGAN-GP loss
        if loss_type == 'WGAN':
            gradient_penalty = calculate_gradient_penalty(
                discriminator, hr_imgs, sr_imgs.detach())
            discriminator_loss = (torch.mean(
                sr_discriminated) - torch.mean(hr_discriminated) + lambda_gp * gradient_penalty)
        else:
            discriminator_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                adversarial_loss_criterion(
                    hr_discriminated, torch.ones_like(hr_discriminated))

        # Backpropagate discriminator loss
        optimizer_d.zero_grad()
        discriminator_loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)  # Optionally clip gradients
        optimizer_d.step()

        # Accumulate losses for logging
        total_content_loss += content_loss.item()
        total_mse_loss += mse_loss.item()
        total_adversarial_loss += adversarial_loss.item()
        total_perceptual_loss += perceptual_loss.item()
        total_discriminator_loss += discriminator_loss.item()

    # Clean up tensors to free memory
    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

    # Return average losses over the epoch
    return {
        'perceptual_loss': total_perceptual_loss / len(train_loader),
        'content_loss': total_content_loss / len(train_loader),
        'adversarial_loss': total_adversarial_loss / len(train_loader),
        'mse_loss': total_mse_loss / len(train_loader),
        'discriminator_loss': total_discriminator_loss / len(train_loader)
    }


if __name__ == '__main__':
    main()
