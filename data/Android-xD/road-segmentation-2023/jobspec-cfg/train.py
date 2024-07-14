import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import Subset

from dataset import CustomImageDataset, split
from models.deeplabv3 import createDeepLabv3
from models.fpn import get_fpn
from models.unet_backbone import get_Unet
from utils.utils import (accuracy_precision_and_recall, aggregate_tile,
                         bce_loss, rich_loss)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train image segmentation network')
    
    # output directory
    parser.add_argument('--out',
                        help='directory to save outputs',
                        default='out',
                        type=str)
    
    # training data directory
    parser.add_argument('--data',
                        help='directory to load data',
                        default='./data/training',
                        type=str)

    # model
    parser.add_argument('--model',
                        help='model',
                        type=str)

    # load model state for pth.tar file
    parser.add_argument('--load_model',
                        help='filepath to load the model out/*.pth.tar or None',
                        default=None,
                        type=str)

    # specify file to store model state
    parser.add_argument('--store_model',
                        help='filepath to store the model',
                        default='model_best.pth.tar',
                        type=str)
        
    # number of epochs
    parser.add_argument('--epochs',
                        help='num of epochs',
                        default=30,
                        type=int)

    # learning rate
    parser.add_argument('--lr',
                        help='learning rate',
                        default=1e-4,
                        type=float)

    # rich labels
    parser.add_argument('--rich',
                        help='rich labels',
                        default=False,
                        type=bool)
   
    # logging frequency
    parser.add_argument('--freq',
                        help='frequency of logging',
                        default=1,
                        type=int)

    # rich labels
    parser.add_argument('--full',
                        help='train on full dataset',
                        default=False,
                        type=bool)

    # rich labels
    parser.add_argument('--augmentations',
                        help='use augmentations or not',
                        default=True,
                        type=bool)
    
    # parse arguments
    args = parser.parse_args()

    return args


def save_checkpoint(state, is_best, output_dir, filename='model_best.pth.tar'):
    """Save model checkpoint

    Args:
        states: model states.
        is_best (bool): whether to save this model as best model so far.
        output_dir (str): output directory to save the checkpoint
        filename (str): checkpoint name
    """
    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    # save the checkpoint
    if is_best:
        torch.save(state, os.path.join(output_dir, filename))


if __name__ == '__main__':
    """
    Train a model

    Usage:
        python train.py --out <output directory> --data <data directory> --model <model name> --epochs <num of epochs> --rich <rich labels> --freq <logging frequency>
    """
    # set the random seed
    torch.manual_seed(0)

    # set the device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # parse arguments
    args = parse_args()

    # initialize tensorboard writer
    run_id = time.strftime("_%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=args.out + f"/logs/{run_id}")

    # set the data directory
    training_set = args.data

    # set the model
    if args.model == 'deeplabv3':
        get_model = createDeepLabv3
    elif args.model == 'unet':
        get_model = get_Unet
    elif args.model == 'fpn':
        get_model = get_fpn
    else:
        raise ValueError('Invalid model name')

    # set the number of epochs
    train_epochs = args.epochs

    # create the model
    output_channels = 5 if args.rich else 1
    model, preprocess, postprocess = get_model(output_channels, 400)
    if not args.load_model is None:
        state_dict = torch.load(args.load_model, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

    # create the dataset
    dataset_aug = CustomImageDataset(training_set, train=True, rich=args.rich, geo_aug=args.augmentations, color_aug=args.augmentations)
    dataset_clean = CustomImageDataset(training_set, train=True, rich=args.rich, geo_aug=False, color_aug=False)

    # split the dataset into train, calibration, and validation sets
    train_indices, cal_indices, val_indices = split(len(dataset_aug), [0.90, 0.05, 0.05])
    train_dataset = Subset(dataset_aug, train_indices)
    if args.full:
        train_dataset = dataset_aug
    cal_dataset = Subset(dataset_clean, cal_indices)
    val_dataset = Subset(dataset_clean, val_indices)

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    # set the loss function
    if args.rich:
        loss_fn = rich_loss
    else:
        loss_fn = bce_loss

    # set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1.0, total_iters=60)


    # Training loop
    best_score = 100
    for epoch in range(train_epochs):
        # train for one epoch
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_precision = 0.0
        train_recall = 0.0
        for i, (input, target) in enumerate(train_loader):
            # Move input and target tensors to the device (CPU or GPU)
            input = input.to(device)
            target = target.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            output = postprocess(model(preprocess(input)))

            # Compute the loss
            loss = loss_fn(output, target)

            # Backward pass
            loss.backward()

            # Accumulate loss
            train_loss += loss.item()

            # binary classification
            y_gt = target[:, :1]
            y_pred = F.sigmoid(output[:, :1])

            # compute accuracy, precision, and recall
            a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred>0.5)*1.))
            train_accuracy += a
            train_precision += b
            train_recall += c

            # Update the parameters
            optimizer.step()

            # Print progress
            if (i + 1) % args.freq == 0:
                print(f'Train Epoch: {epoch + 1} [{i + 1}/{len(train_loader)}]\t'
                      f'Loss: {train_loss / (i + 1):.4f}\t'
                      f'Accuracy: {train_accuracy/(i+1):.4f}')

        # update learning rate
        scheduler.step()
        
        # compute metrics
        train_accuracy /= i + 1
        train_loss /= i + 1
        train_recall /= i+1
        train_precision /= i+1
        train_f1 = 1/(1/train_recall + 1/train_precision)
        
        # Calibration
        cal_loss = 0.0
        cal_accuracy = 0.0
        cal_precision = 0.0
        cal_recall = 0.0

        # evaluate the model on the calibration and validation set with aggregation
        model.eval()
        with torch.no_grad():
            # calibrate thresholds on calibration split
            m = len(cal_loader)
            n_ticks = 101
            recall_space_16 = torch.zeros((m, n_ticks, n_ticks))
            precision_space_16 = torch.zeros((m, n_ticks, n_ticks))
            ticks = torch.linspace(0, 1, n_ticks)

            # evaluation loop
            for i, (input, target) in enumerate(cal_loader):
                # Move input and target tensors to the device (CPU or GPU)
                input = input.to(device)
                target = target.to(device)

                # Forward pass
                output = postprocess(model(preprocess(input)))
                y_pred = F.sigmoid(output[:, :1])
                y_gt = target[:, :1]

                # aggregation loss
                agg_target = aggregate_tile(target[:, :1])
                for r, th1 in enumerate(ticks):
                    for c, th2 in enumerate(ticks):
                        _, precision_space_16[i, r, c], recall_space_16[i, r, c] = \
                            accuracy_precision_and_recall(agg_target, aggregate_tile((y_pred > th1) * 1.0, thresh=th2))

                # Compute accuracy, precision, and recall
                a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred > 0.5) * 1.))
                cal_accuracy += a
                cal_precision += b
                cal_recall += c

                # Print progress for calibration
                if (i + 1) % args.freq == 0:
                    print(f'Calibration Epoch: {epoch + 1} [{i + 1}/{len(cal_loader)}]\t'
                          f'Loss: {cal_loss / (i + 1):.4f}\t'
                          f'Accuracy: {cal_accuracy / (i + 1):.4f}')

            # compute calibration metrics
            recall_space_16 = torch.mean(recall_space_16, dim=0)
            precision_space_16 = torch.mean(precision_space_16, dim=0)
            f1_space = 2. / (1 / recall_space_16 + 1 / precision_space_16)
            cal_f1_cal = torch.max(f1_space)
            loc = (f1_space == torch.max(f1_space)).nonzero()[0]
            cal_recall_cal = recall_space_16[loc[0],loc[1]]
            cal_precision_cal = precision_space_16[loc[0],loc[1]]
            th1, th2 = ticks[loc[0]], ticks[loc[1]]
            cal_accuracy /= i + 1
            cal_precision /= i + 1
            cal_recall /= i+1
            cal_f1 = 2/(1/train_recall + 1/train_precision)

            # Print calibration scores
            print(f'Calibration: \t'
                  f'F1 Uncalibrated: {cal_f1:.4f}\t'
                  f'F1 Calibrated: {cal_f1_cal:.4f}\t'
                  f'Thresholds: ({th1:.2f},{th2:.2f})')

            print(
                f'Uncalibrated \tLoss: {cal_loss:.4f}\t F1: {cal_f1:.4f} \t Accuracy: {cal_accuracy:.4f}\t'
                f'Recall: {cal_recall:.4f}\t Precision: {cal_precision:.4f}')
            print(
                f'Calibrated \tLoss: {cal_loss:.4f}\t F1: {cal_f1_cal:.4f} \t Accuracy: ---- \t'
                f'Recall: {cal_recall_cal:.4f}\t Precision: {cal_precision_cal:.4f}')

            # validation
            val_loss = 0.0
            val_recall = 0.0
            val_precision = 0.0
            val_accuracy = 0.0
            val_recall_cal = 0.0
            val_precision_cal = 0.0
            val_accuracy_cal = 0.0
            for i, (input, target) in enumerate(val_loader):
                # Move input and target tensors to the device (CPU or GPU)
                input = input.to(device)
                target = target.to(device)

                y_gt = target[:, :1]
                # Forward pass
                output = postprocess(model(preprocess(input)))

                # Compute the loss
                loss = loss_fn(output, target)

                # Accumulate loss
                val_loss += loss.item()

                # binary classification
                y_pred = F.sigmoid(output[:, :1])
                pred = (y_pred > 0.5)*1.

                # Compute accuracy, precision, and recall with the uncalibrated thresholds
                a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred > 0.5) * 1.))
                val_accuracy += a
                val_precision += b
                val_recall += c

                # Compute accuracy, precision, and recall with the calibrated thresholds
                a, b, c = accuracy_precision_and_recall(aggregate_tile(y_gt), aggregate_tile((y_pred > th1) * 1., th2))
                val_accuracy_cal += a
                val_precision_cal += b
                val_recall_cal += c
   
            # compute metrics
            val_loss /= i + 1
            val_accuracy /= i + 1
            val_recall /= i + 1
            val_precision /= i + 1
            val_accuracy_cal /= i + 1
            val_recall_cal /= i + 1
            val_precision_cal /= i + 1
            val_f1 = 2/(1/val_recall + 1/val_precision)
            val_f1_cal = 2/(1/val_recall_cal + 1/val_precision_cal)

        # save the model if it is the best so far
        is_best = val_loss < best_score
        if is_best:
            best_score = val_loss
        save_checkpoint(model.state_dict(), is_best, args.out, args.store_model)

        # Print progress
        print(f'Validation Epoch: {epoch + 1}\tLoss: {val_loss:.4f}\t F1: {val_f1:.4f} \t Accuracy: {val_accuracy:.4f}\t'
              f'Recall: {val_recall:.4f}\t Precision: {val_precision:.4f}')
        print(f'Val Calibr Epoch: {epoch + 1}\tLoss: {val_loss:.4f}\t F1: {val_f1_cal:.4f} \t Accuracy: {val_accuracy_cal:.4f}\t'
              f'Recall: {val_recall_cal:.4f}\t Precision: {val_precision_cal:.4f}')
        
        # log metrics
        writer.add_scalars("Loss", {"val": val_loss, "train": train_loss}, epoch)
        writer.add_scalar("F1/val", val_f1 / (i + 1), epoch)
        writer.add_scalars("Accuracy", {"val": val_accuracy, "train": train_accuracy}, epoch)


 