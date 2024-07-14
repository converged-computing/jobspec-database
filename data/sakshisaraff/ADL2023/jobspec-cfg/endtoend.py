#!/usr/bin/env python3

import argparse
import io
from pathlib import Path
from multiprocessing import cpu_count
import os
import numpy as np
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import dataset
from Trainer import Trainer
from CNN import CNN
from CNN_extension import CNN_extension
from SampleCNN import SampleCNN

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.backends.cudnn.benchmark = True

#Parser arguments
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser = argparse.ArgumentParser(
    description="Train a simple CNN on MagnaTagATune for End-to-End Learning",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--momentum", default=0.95, type=float)
parser.add_argument("--normalisation", default="minmax", type=str, help="minmax or standardisation")
parser.add_argument(
    "--length-conv",
    default=256,
    type=int,
    help="Length used in the Stride Convolution Layer",
)
parser.add_argument(
    "--stride-conv",
    default=256,
    type=int,
    help="Stride used in the Stride Convolution Layer",
)
parser.add_argument(
    "--model",
    default="Basic",
    type=str,
    help="Basic or Extension1 or Deep",
)
parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of audio examples within each batch",
)
parser.add_argument(
    "--outchannel-stride",
    default=32,
    type=int,
    help="Out channels in the Stride Convolution",
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--eval-frequency",
    default=2,
    type=int,
    help="How frequently to evaluate the model in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--mode",
    default="training",
    help="Choices are hyperparameter-tuning, or training",
)
parser.add_argument(
    "--inner-norm",
    default="Group",
    help="None or Layer (Layer Normalisation) or Batch (Batch Normalisation) or Group (Group Normalisation)",
)


def main(args):
    path_annotations_train = Path("annotations/train_labels.pkl")
    path_annotations_val = Path("annotations/val_labels.pkl")
    path_annotations_test = Path("annotations/test_labels.pkl")
    train_dataset = dataset.MagnaTagATune(path_annotations_train, Path("samples/"))
    val_dataset = dataset.MagnaTagATune(path_annotations_val, Path("samples/"))
    test_dataset = dataset.MagnaTagATune(path_annotations_test, Path("samples/"))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    # unshuffled training dataset for auc for training curve 
    train_auc = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # Values are the same for all datasets, hence we don't recalculate them for different contexts.
    minval, maxval = minmax(train_loader)
    
    
    ## code to generate the number of samples of each class in train, val, test
    ## uncomment to print out in log file
    #tr_samples_count = sample_count(train_loader)
    #val_samples_count = sample_count(val_loader)    
    # test_samples_count = sample_count(test_loader)
    # print("Train Dataset Specifics: ")
    # print(tr_samples_count)
    # print("Validation Dataset Specifics: ")
    # print(val_samples_count)
    # print("Test Dataset Specifics: ")
    # print(test_samples_count)

    trainer = None
    model_path = None
    # Ideally we make this from the command line args
    # We can iterate through args
    # How do we only include the args that relate to hyperparameters though?
    # Right now when we add a hyperparameter we need to:
        # Add it to hyperparameter_possibilities
        # Add it to hyperparameter_choices REMOVE
        # Add it to args
        # Pass it to the right thing in train() REMOVE
        # Make use of it in the correct place
    # Ideally, we just want to:
        # Add it to args
        # Make use of it in the correct place
        # Add it to hyperparameter_possibilities if we intend to hyperparameter tune it
    hyperparameter_choices = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "dropout": args.dropout,
        "length_conv": args.length_conv,
        "stride_conv": args.stride_conv,
        "normalisation": args.normalisation,
        "outchannel_stride": args.outchannel_stride,
        "model": args.model,
        "inner_norm": args.inner_norm,
    }
    """
    If you specify on the command line that you want to tune hyperparameters,
    use grid search to alter hyperparameter_choices.
    """
    if args.mode == "hyperparameter-tuning":
        scored_hyperparameter_choices = []
        hyperparameter_possibilities = {
            #"batch_size": [],
            "learning_rate": [0.015, 0.0075],
            #"normalisation": ["minmax", "Sakshi"]
            #"learning_rate": [0.0005, 0.001, 0.0015, 0.005, 0.01],
            "momentum": [0.92, 0.95, 0.97],
            #"epochs": [],
            #"dropout": [],
            #"length_conv": [],
            #"stride_conv": []
        }
        for hyperparameter, possibilities in hyperparameter_possibilities.items():
            best_auc = 0
            best_choice = None
            for possibility in possibilities:
                if hyperparameter == "batch_size":
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size=possibility,
                        pin_memory=True,
                        num_workers=args.worker_count,
                    )
                hyperparameter_choices[hyperparameter] = possibility
                trainer, model, model_path = train(
                    args,
                    minval,
                    maxval,
                    train_loader,
                    train_auc,
                    val_loader,
                    args.eval_frequency,
                    args.print_frequency,
                    args.log_frequency,
                    hyperparameter_choices,
                    path_annotations_val,
                )
                auc = trainer.evaluate(path_annotations_val, val_loader)
                scored_hyperparameter_choices.append((auc, hyperparameter_choices.copy()))
                if auc > best_auc:
                    best_choice = possibility
                    best_auc = auc
                print("Test Results on Best Model (Based on val AUC):")
                trainer.test_evaluate(path_annotations_test, model_path, test_loader)

            """
            Once we've tested all the different batch sizes,
            we can make a loader with the best one.
            """
            if hyperparameter == "batch_size":
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=True,
                    batch_size=hyperparameter_choices["batch_size"],
                    pin_memory=True,
                    num_workers=args.worker_count,
                )
            hyperparameter_choices[hyperparameter] = best_choice
        print("Hyperparameter tuning done")
        print(scored_hyperparameter_choices)
    else:
        trainer, model, model_path = train(
            args,
            minval,
            maxval,
            train_loader,
            train_auc,
            val_loader,
            args.eval_frequency,
            args.print_frequency,
            args.log_frequency,
            hyperparameter_choices,
            path_annotations_val
        )
    
    print("Evaluating against test data...")
    # print(hyperparameter_choices)
    print("Test Results on Final Model:")
    trainer.evaluate(path_annotations_test, test_loader, hyperparameter_choices["epochs"] - 1)
    print("Test Results on Best Model (Based on val AUC):")
    trainer.test_evaluate(path_annotations_test, model_path, test_loader)

def get_summary_writer_log_dir(args: argparse.Namespace, hyperparameters) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = io.StringIO()
    tb_log_dir_prefix.write(f'CNN_')
    # for hyperparameter, value in hyperparameters.items():
    #     tb_log_dir_prefix.write(f'{hyperparameter}={value}_')
    if args.model == "Basic":
        tb_log_dir_prefix.write(f'basic_')
        if args.length_conv != 256 and args.length_conv != 512:
            tb_log_dir_prefix.write(f'length_conv={args.length_conv}_stride_conv={args.stride_conv}')
    elif args.model == "Extension1":
        tb_log_dir_prefix.write(f'extension1_')
    elif args.model == "Deep":
        tb_log_dir_prefix.write(f'deep_')
    tb_log_dir_prefix.write(f'run_')
    tb_log_dir_prefix = tb_log_dir_prefix.getvalue()
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def train(
        args,
        minval,
        maxval,
        train_loader,
        train_unshuffled,
        inter_eval_loader,
        eval_frequency,
        print_frequency,
        log_frequency,
        hyperparameters,
        sample_path
):
    log_dir = get_summary_writer_log_dir(args, hyperparameters)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )
    print("Start of Training")
    #print("Hyperparameters: " + str(hyperparameters))
    #calls the different models coded into the project
    if args.model == "Basic":
        ## uncomment to allow for changing parameters and hyperparameters
        # model = CNN(
        #     length=hyperparameters["length_conv"],
        #     stride=hyperparameters["stride_conv"],
        #     channels=1,
        #     class_count=50,
        #     minval=minval,
        #     maxval=maxval,
        #     normalisation=hyperparameters["normalisation"],
        #     out_channels=hyperparameters["outchannel_stride"]
        # )
        #optimizer = optim.SGD(model.parameters(), lr=hyperparameters["learning_rate"], momentum=hyperparameters["momentum"])
        model = CNN(
            length=hyperparameters["length_conv"],
            stride=hyperparameters["stride_conv"],
            channels=1,
            class_count=50,
            minval=minval,
            maxval=maxval,
            normalisation="minmax",
            out_channels=32
        )
        optimizer = optim.SGD(model.parameters(), lr=0.0075, momentum=0.95)
        scheduler = None
    elif args.model == "Extension1":
        ## uncomment to allow for changing parameters and hyperparameters
        # model = CNN_extension(
        #     length=hyperparameters["length_conv"],
        #     stride=hyperparameters["stride_conv"],
        #     channels=1,
        #     class_count=50,
        #     minval=minval,
        #     maxval=maxval,
        #     normalisation=hyperparameters["normalisation"],
        #     out_channels=hyperparameters["outchannel_stride"],
        #     dropout=hyperparameters["dropout"],
        #     inner_norm=hyperparameters["inner_norm"],
        # )
        model = CNN_extension(
            length=256,
            stride=256,
            channels=1,
            class_count=50,
            minval=minval,
            maxval=maxval,
            normalisation="minmax",
            out_channels=32,
            dropout=0.2,
            inner_norm="Group",
        )
        optimizer = optim.SGD(model.parameters(), lr=0.0075, momentum=0.95)
        scheduler = None
        print("no scheduler")
    elif args.model == "Deep":
        model = SampleCNN(
            class_count=50
        )
        optimizer = optim.SGD(model.parameters(), lr=0.01,
                              momentum=0.95, nesterov=True)
        scheduler = None
        print("no scheduler")
    criterion = nn.BCELoss()
    trainer = Trainer(
        model,
        train_loader,
        train_unshuffled,
        inter_eval_loader,
        criterion,
        optimizer,
        scheduler,
        summary_writer,
        DEVICE
    )
    if not Path("models//").is_dir():
        os.mkdir(Path("models//"))
    model_path = Path("models//" + log_dir[5:])
    print(f"Writing model to {model_path}")
    trainer.train(
        hyperparameters,
        sample_path,
        model_path,
        hyperparameters["epochs"],
        eval_frequency,
        print_frequency=print_frequency,
        log_frequency=log_frequency,
    )
    summary_writer.close()
    return trainer, model, model_path

##used for input normalisation- finds the min and max of the dataloader parameter
def minmax(dataloader):
    maxval = float("-inf")
    minval = float("inf")
    for filename, data, label in dataloader:
        if minval > torch.min(data):
            minval = torch.min(data)
        if maxval < torch.max(data):
            maxval = torch.max(data)
    return minval, maxval

#counts the number of samples within each label
def sample_count(data_loader: DataLoader):
    l_counts = {}
    for i in range(50):
        key = i
        l_counts[key] = 0
    for filename, batch, labels in data_loader:
        labels = labels.cpu().numpy()
        for l in labels:
            index = np.where(l == 1)
            for i in index[0]:
                print(index)
                l_counts[i] += 1
    return l_counts
        
if __name__ == "__main__":
    main(parser.parse_args())
