# Imports
import os
import argparse
import numpy as np
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.utils.class_weight import compute_class_weight

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

# Weights and Biases (W&B) Imports
import wandb

# Log in to W&B Account
wandb.login()

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Project Imports
from data_utilities import CUB2002011Dataset, PAPILADataset, PH2Dataset, STANFORDCARSDataset
from model_utilities import DenseNet, ResNet, VGG
from train_val_test_utilities import model_train, model_validation, last_only, warm_only, joint, print_metrics



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--data_dir', type=str, default="data", help="Directory of the data set.")

# Data set
parser.add_argument('--dataset', type=str, required=True, choices=["CUB2002011", "PAPILA", "PH2", "STANFORDCARS"], help="Data set: CUB2002011, PAPILA, PH2, STANFORDCARS.")

# Model
parser.add_argument('--base_architecture', type=str, required=True, choices=["densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"], help='Base architecture: densenet121, densenet161, resnet34, resnet152, vgg16, vgg19.')

# Batch size
parser.add_argument('--batchsize', type=int, default=4, help="Batch-size for training and validation")

# Image size
# img_size = 224
parser.add_argument('--img_size', type=int, default=224, help="Size of the image after transforms")


# Joint optimizer learning rates
# joint_optimizer_lrs = {'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}
parser.add_argument('--joint_optimizer_lrs', type=dict, default={'features': 1e-4}, help="Joint optimizer learning rates.")

# Joint learning rate step size
# joint_lr_step_size = 5
parser.add_argument('--joint_lr_step_size', type=int, default=5, help="Joint learning rate step size.")

# Last layer optimizer learning rate
# last_layer_optimizer_lr = 1e-4
parser.add_argument('--last_layer_optimizer_lr', type=float, default=1e-4, help="Last layer optimizer learning rate.")

# Class Weights
parser.add_argument("--classweights", action="store_true", help="Weight loss with class imbalance")

# Number of training epochs
# num_train_epochs = 1000
parser.add_argument('--num_train_epochs', type=int, default=50, help="Number of training epochs.")

# Number of warm epochs
# num_warm_epochs = 5
parser.add_argument('--num_warm_epochs', type=int, default=5, help="Number of warm epochs.")

# Push start
# push_start = 10
parser.add_argument('--push_start', type=int, default=10, help="Push start.")

# Output directory
parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")

# Number of workers
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")

# GPU ID
parser.add_argument("--gpu_id", type=int, default=0, help="The index of the GPU")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint from which to resume training")



# Parse the arguments
args = parser.parse_args()


# Resume training
RESUME = args.resume

# Training checkpoint
if RESUME:
    CHECKPOINT = args.checkpoint

    assert CHECKPOINT is not None, "Please specify the model checkpoint when resume is True"


# Data directory
DATA_DIR = args.data_dir

# Dataset
DATASET = args.dataset

# Base Architecture
BASE_ARCHITECTURE = args.base_architecture

# Results Directory
OUTPUT_DIR = args.output_dir

# Number of workers (threads)
WORKERS = args.num_workers

# Number of training epochs
NUM_TRAIN_EPOCHS = args.num_train_epochs

# Number of warm epochs
NUM_WARM_EPOCHS = args.num_warm_epochs

# Push start
PUSH_START = args.push_start

# Push epochs
# push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
PUSH_EPOCHS = [i for i in range(NUM_TRAIN_EPOCHS) if i % 10 == 0]

# Joint optimizer learning rates
JOINT_OPTIMIZER_LRS = args.joint_optimizer_lrs

# JOINT_LR_STEP_SIZE
JOINT_LR_STEP_SIZE = args.joint_lr_step_size

# LAST_LAYER_OPTIMIZER_LR
LAST_LAYER_OPTIMIZER_LR = args.last_layer_optimizer_lr

# Batch size
BATCH_SIZE = args.batchsize

# Image size (after transforms)
IMG_SIZE = args.img_size



# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(OUTPUT_DIR, DATASET.lower(), "baseline", BASE_ARCHITECTURE.lower(), timestamp)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# Save training parameters
with open(os.path.join(results_dir, "train_params.txt"), "w") as f:
    f.write(str(args))



# Set the W&B project
wandb.init(
    project="proto-counterfactuals", 
    name=timestamp,
    config={
        "architecture": "baseline-" + BASE_ARCHITECTURE.lower(),
        "dataset": DATASET.lower(),
        "train-epochs": NUM_TRAIN_EPOCHS,
    }
)


# Load data
# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Input Data Dimensions
img_height = IMG_SIZE
img_width = IMG_SIZE


# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Validation Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


# Dataset
# CUB2002011
if DATASET == "CUB2002011":
    # Train Dataset
    train_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "train", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=True,
        transform=train_transforms
    )

    # Validation Dataset
    val_set = CUB2002011Dataset(
        data_path=os.path.join(DATA_DIR, "cub2002011", "processed_data", "test", "cropped"),
        classes_txt=os.path.join(DATA_DIR, "cub2002011", "source_data", "classes.txt"),
        augmented=False,
        transform=val_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.labels_dict)



# PAPILA
elif DATASET == "PAPILA":
    # Train Dataset
    train_set = PAPILADataset(
        data_path=DATA_DIR,
        subset="train",
        cropped=True,
        augmented=True,
        transform=train_transforms
    )

    # Validation Dataset
    val_set = PAPILADataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        augmented=False,
        transform=val_transforms
    )

    # Number of Classes
    NUM_CLASSES = len(np.unique(train_set.images_labels))



# PH2
elif DATASET == "PH2":
    # Train Dataset
    train_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="train",
        cropped=True,
        augmented=True,
        transform=train_transforms
    )

    # Validation Dataset
    val_set = PH2Dataset(
        data_path=DATA_DIR,
        subset="test",
        cropped=True,
        augmented=False,
        transform=val_transforms
    )

    # Number of Classes
    NUM_CLASSES = len(train_set.diagnosis_dict)



# STANFORDCARS
elif DATASET == "STANFORDCARS":
    # Train Dataset
    train_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_train",
        augmented=True,
        cropped=True,
        transform=train_transforms
    )

    # Validation Dataset
    val_set = STANFORDCARSDataset(
        data_path=DATA_DIR,
        cars_subset="cars_test",
        augmented=False,
        cropped=True,
        transform=val_transforms
    )

    # Number of classes
    NUM_CLASSES = len(train_set.class_names)



# Results and Weights
weights_dir = os.path.join(results_dir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(results_dir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(results_dir, "tensorboard"), flush_secs=30)


# Choose GPU
DEVICE = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")






# Construct the Model
if BASE_ARCHITECTURE.lower() in ("densenet121", "densenet161"):
    baseline_model = DenseNet(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)
elif BASE_ARCHITECTURE.lower() in ("resnet34", "resnet152"):
    baseline_model = ResNet(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)
else:
    baseline_model = VGG(backbone=BASE_ARCHITECTURE.lower(), channels=3, height=IMG_SIZE, width=IMG_SIZE, nr_classes=NUM_CLASSES)



# Define optimizers and learning rate schedulers
# Joint Optimizer Specs
joint_optimizer_specs = [

    {'params': baseline_model.features.parameters(), 'lr': JOINT_OPTIMIZER_LRS['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': baseline_model.last_layer.parameters(), 'lr': LAST_LAYER_OPTIMIZER_LR},

]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=JOINT_LR_STEP_SIZE, gamma=0.1)


# Warm Optimizer Learning Rates
warm_optimizer_specs = [
    {'params': baseline_model.last_layer.parameters(), 'lr': LAST_LAYER_OPTIMIZER_LR}
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)


# Last Layer Optimizer
last_layer_optimizer_specs = [
    {'params': baseline_model.last_layer.parameters(), 'lr': LAST_LAYER_OPTIMIZER_LR}
]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)


# Put model into DEVICE (CPU or GPU)
baseline_model = baseline_model.to(DEVICE)


# Get model summary
try:
    model_summary = summary(baseline_model, (1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)

except:
    model_summary = str(baseline_model)


# Write into file
with open(os.path.join(results_dir, "model_summary.txt"), 'w') as f:
    f.write(str(model_summary))


# Log model's parameters and gradients to W&B
wandb.watch(baseline_model)


# TODO: Review
# Class weights for loss
# if args.classweights:
#     classes = np.array(range(NUM_CLASSES))
#     cw = compute_class_weight('balanced', classes=classes, y=np.array(train_set.images_labels))
#     cw = torch.from_numpy(cw).float().to(DEVICE)
#     print(f"Using class weights {cw}")
# else:
#     cw = None




# Resume training from given checkpoint
if RESUME:
    checkpoint = torch.load(CHECKPOINT)
    baseline_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    warm_optimizer.load_state_dict(checkpoint['warm_optimizer_state_dict'])
    joint_optimizer.load_state_dict(checkpoint['joint_optimizer_state_dict'])
    init_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from {CHECKPOINT} at epoch {init_epoch}")

else:
    init_epoch = 0


# Dataloaders
# Train
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=WORKERS)

# Validation
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=WORKERS)



# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
print(f'Size of training set: {len(train_loader.dataset)}')
print(f'Size of validation set: {len(val_loader.dataset)}')
print(f'Batch size: {BATCH_SIZE}')



# Define some extra paths and directories
# Directory for saved prototypes
saved_prototypes_dir = os.path.join(weights_dir, 'prototypes')
if not os.path.isdir(saved_prototypes_dir):
    os.makedirs(saved_prototypes_dir)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((NUM_TRAIN_EPOCHS, ))
val_losses = np.zeros_like(train_losses)

# Initialise metrics arrays
train_metrics = np.zeros((NUM_TRAIN_EPOCHS, 4))
val_metrics = np.zeros_like(train_metrics)

# Initialise best accuracy
best_accuracy = -np.inf
best_accuracy_push = -np.inf
best_accuracy_push_last = -np.inf


# Go through the number of Epochs
for epoch in range(init_epoch, NUM_TRAIN_EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Phase
    print("Training Phase")


    if epoch < NUM_WARM_EPOCHS:
        print("Training: Warm Phase")
        warm_only(model=baseline_model)
        metrics_dict = model_train(model=baseline_model, dataloader=train_loader, device=DEVICE, optimizer=warm_optimizer)


    else:
        print("Training: Joint Phase")
        joint(model=baseline_model)
        joint_lr_scheduler.step()
        metrics_dict = model_train(model=baseline_model, dataloader=train_loader, device=DEVICE, optimizer=joint_optimizer)


    # Print metrics
    print_metrics(metrics_dict=metrics_dict)

    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = metrics_dict["run_avg_loss"]
    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_tr_losses.npy")
    np.save(file=fname, arr=train_losses, allow_pickle=True)


    # Train Metrics
    # Acc
    train_metrics[epoch, 0] = metrics_dict['accuracy']
    # Recall
    train_metrics[epoch, 1] = metrics_dict['recall']
    # Precision
    train_metrics[epoch, 2] = metrics_dict['precision']
    # F1-Score
    train_metrics[epoch, 3] = metrics_dict['f1']

    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", metrics_dict["run_avg_loss"], global_step=epoch)
    tbwritter.add_scalar("acc/train", metrics_dict['accuracy'], global_step=epoch)
    tbwritter.add_scalar("rec/train", metrics_dict['recall'], global_step=epoch)
    tbwritter.add_scalar("prec/train", metrics_dict['precision'], global_step=epoch)
    tbwritter.add_scalar("f1/train", metrics_dict['f1'], global_step=epoch)



    # Log to W&B
    wandb_tr_metrics = {
        "loss/train":metrics_dict["run_avg_loss"],
        "acc/train":metrics_dict['accuracy'],
        "rec/train":metrics_dict['recall'],
        "prec/train":metrics_dict['precision'],
        "f1/train":metrics_dict['f1'],
        "epoch/train":epoch
    }
    wandb.log(wandb_tr_metrics)




    # Validation Phase 
    print("Validation Phase")
    metrics_dict = model_validation(model=baseline_model, dataloader=val_loader, device=DEVICE)

    # Print metrics
    print_metrics(metrics_dict=metrics_dict)

    # Append values to the arrays
    # Validation Loss
    val_losses[epoch] = metrics_dict["run_avg_loss"]
    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_val_losses.npy")
    np.save(file=fname, arr=val_losses, allow_pickle=True)


    # Train Metrics
    # Acc
    val_metrics[epoch, 0] = metrics_dict['accuracy']
    # Recall
    val_metrics[epoch, 1] = metrics_dict['recall']
    # Precision
    val_metrics[epoch, 2] = metrics_dict['precision']
    # F1-Score
    val_metrics[epoch, 3] = metrics_dict['f1']

    # Save it to directory
    fname = os.path.join(history_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_val_metrics.npy")
    np.save(file=fname, arr=val_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/val", metrics_dict["run_avg_loss"], global_step=epoch)
    tbwritter.add_scalar("acc/val", metrics_dict['accuracy'], global_step=epoch)
    tbwritter.add_scalar("rec/val", metrics_dict['recall'], global_step=epoch)
    tbwritter.add_scalar("prec/val", metrics_dict['precision'], global_step=epoch)
    tbwritter.add_scalar("f1/val", metrics_dict['f1'], global_step=epoch)



    # Log to W&B
    wandb_val_metrics = {
        "loss/val":metrics_dict["run_avg_loss"],
        "acc/val":metrics_dict['accuracy'],
        "rec/val":metrics_dict['recall'],
        "prec/val":metrics_dict['precision'],
        "f1/val":metrics_dict['f1'],
        "epoch/val":epoch
    }
    wandb.log(wandb_val_metrics)



    # save.save_model_w_condition(model=ppnet_model, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu, target_accu=0.70, log=log)
    # Save checkpoint
    if metrics_dict['accuracy'] > best_accuracy:

        print(f"Accuracy increased from {best_accuracy} to {metrics_dict['accuracy']}. Saving new model...")

        # Model path
        model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best.pt")
        save_dict = {
            'epoch':epoch,
            'model_state_dict':baseline_model.state_dict(),
            'warm_optimizer_state_dict':warm_optimizer.state_dict(),
            'joint_optimizer_state_dict':joint_optimizer.state_dict(),
            'run_avg_loss': metrics_dict["run_avg_loss"],
        }
        torch.save(save_dict, model_path)
        print(f"Successfully saved at: {model_path}")

        # Update best accuracy value
        best_accuracy = metrics_dict['accuracy']



    # PUSH START and PUSH EPOCHS
    if epoch >= PUSH_START and epoch in PUSH_EPOCHS:

        print("Optimizing last layer...")
        last_only(model=baseline_model)
        
        for i in range(20):
            print(f'Step {i+1} of {20}')
            print("Training")
            metrics_dict = model_train(model=baseline_model, dataloader=train_loader, device=DEVICE, optimizer=last_layer_optimizer)
            print_metrics(metrics_dict=metrics_dict)
            
            print("Validation")
            metrics_dict = model_validation(model=baseline_model, dataloader=val_loader, device=DEVICE)
            print_metrics(metrics_dict=metrics_dict)

            # Save checkpoint
            if metrics_dict['accuracy'] > best_accuracy_push_last:

                print(f"Accuracy increased from {best_accuracy_push_last} to {metrics_dict['accuracy']}. Saving new model...")

                # Model path
                model_path = os.path.join(weights_dir, f"{BASE_ARCHITECTURE.lower()}_{DATASET.lower()}_best_push_last.pt")
                save_dict = {
                    'epoch':epoch,
                    'model_state_dict':baseline_model.state_dict(),
                    'warm_optimizer_state_dict':warm_optimizer.state_dict(),
                    'joint_optimizer_state_dict':joint_optimizer.state_dict(),
                    'run_avg_loss': metrics_dict["run_avg_loss"],
                }
                torch.save(save_dict, model_path)
                print(f"Successfully saved at: {model_path}")

                # Update best accuracy value
                best_accuracy_push_last = metrics_dict['accuracy']



# Finish statement and W&B
wandb.finish()
print("Finished.")
