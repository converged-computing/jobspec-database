import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import wandb
import os
import argparse

from dataloader.dataloaders import get_dataloader, get_numclass

parser = argparse.ArgumentParser()
parser.add_argument("-r","--data_root", default = "./Data", help="Directory to which dataset should be downloaded", type=str)
parser.add_argument("--model_root", default = "./Models", help="Directory to which models should be saved", type=str)
parser.add_argument("-d","--dataset", required = True, help="desired dataset", type=str, 
                    choices=["Cars","DTD","MNIST","iWildCam","GTSRB","EuroSAT","Resisc45","SUN397","SVHN","Caltech256","CIFAR10","fmow","rxrx1"])
parser.add_argument("-n","--num_workers", default=1, help="number of workers needed", type=int)
parser.add_argument("--batch_size", default=32, help="Specift batch size for dataloaders", type=int)
parser.add_argument("-e","--num_epochs", default=80, type=int)
parser.add_argument("--log_interval", default=100, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("-wd","--weight_decay", default=0.0, type=float)
parser.add_argument("-o","--optimizer", default="ADAM", type=str, choices = ["ADAM", "SGD"])
parser.add_argument("-lrs","--lr_scheduler", default="EXP", type=str, choices = ["EXP", "STEP","NONE"])
parser.add_argument("-g","--gamma_for_lr_scheduler", default = 0.95, type=float)
parser.add_argument("-ss","--step_size_lr_scheduler", default = 20, type=float)
parser.add_argument("-mom","--momentum", default=0.9, type=float, help="momentum is only used when we use SGD as optimizer")
parser.add_argument("-m","--model_name", required = True, choices = ["Resnet18", "Resnet50", "Resnet101"],type=str)
parser.add_argument("-pn","--wandb_project_name", required = True, type=str)


args = parser.parse_args()

dataset_root = args.data_root
dataset = args.dataset
num_workers = args.num_workers
num_epochs = args.num_epochs
log_interval = args.log_interval
lr=args.lr
batch_size = args.batch_size
model_name = args.model_name
model_root = args.model_root
weight_decay = args.weight_decay
opt = args.optimizer
gamma = args.gamma_for_lr_scheduler
lrs = args.lr_scheduler
ss = args.step_size_lr_scheduler
momentum = args.momentum
model_path = model_root+"/"+model_name+"-"+dataset+"-"+opt+".pth"

os.makedirs(dataset_root, exist_ok=True)
os.makedirs(model_root, exist_ok=True)
num_classes = get_numclass(dataset)

os.environ["WANDB__SERVICE_WAIT"] = "300"
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project=args.wandb_project_name,
    name=model_name+"-"+opt,
    config={
    "learning_rate": lr,
    "architecture": model_name,
    "dataset": dataset,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "optimizer": opt,
    "weight_decay": weight_decay,
    "lr_scheduler": lrs,
    "stepsize": ss
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device is {device}")

# ========== Load Data =========== 
train_dataloader, val_dataloader, test_dataloader = get_dataloader(ds=dataset,root=dataset_root,bs=batch_size,nworkers=num_workers)

# ========== Get Pretrained Model =========== 
if(model_name=="Resnet18"):
    model = models.resnet18(weights='IMAGENET1K_V1')
elif(model_name=="Resnet50"):
    model = models.resnet50(weights='IMAGENET1K_V2')
elif(model_name=="Resnet101"):
    model = models.resnet101(weights='IMAGENET1K_V2')
    
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Swap fc layer for finetune dataset
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# =========== Define Optimizer =============
def get_optimizer(model, lr):
    if(opt=="ADAM"):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    elif(opt=="SGD"):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer
optimizer = get_optimizer(model,lr)
curr_lr = lr
    
# ============ Define Learning Rate Decay Scheduler ==========
if(lrs=="STEP"):
    print(f"Using {lrs} LR-Scheduler with {ss=} and {gamma=}")
    scheduler = StepLR(optimizer, step_size=ss, gamma=gamma)
elif(lrs == "EXP"):
    print(f"Using {lrs} LR-Scheduler with {gamma=}")
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
# =========== Define Loss Function =============
loss_function = nn.CrossEntropyLoss()

# ========== Fully Fine-Tune Model =========== 
model.train()
model = nn.DataParallel(model)
model = model.to(device)
print(f"model is moved to {device}")

max_acc = 0
acc_threshold = 0.05
min_loss = 10
# ========== Training Routine ============
for epoch in range(num_epochs):
    # =========== Layer by Layer Unfreezing =========
    if(epoch == ss):
        for param in model.module.layer4.parameters():
            param.requires_grad = True
        curr_lr = lr*0.2
        optimizer = get_optimizer(model,curr_lr)
        print("unfreezing layer 4")
    if(epoch == 2*ss):
        for param in model.module.layer3.parameters():
            param.requires_grad = True
        curr_lr = lr*0.1
        optimizer = get_optimizer(model,curr_lr)
        print("unfreezing layer 3")
    if(epoch == 3*ss):
        for param in model.module.layer2.parameters():
            param.requires_grad = True
        urr_lr = lr*0.05
        optimizer = get_optimizer(model,curr_lr)
        print("unfreezing layer 2")
    if(epoch == 4*ss):
        for param in model.parameters():
            param.requires_grad = True
        urr_lr = lr*0.01
        optimizer = get_optimizer(model,curr_lr)
        print("unfreezing entire model")

    total = 0
    correct = 0
    for batch_idx, labeled_batch in enumerate(train_dataloader):
        # Unpack Batch for iwildcam is slightly different since there's extra metadata
        if (dataset=="iWildCam" or dataset=="rxrx1" or dataset=="fmow"):
            data, targets, metadata = labeled_batch
        else: 
            data, targets = labeled_batch
        data, targets = data.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(data)
        loss = loss_function(outputs, targets)
        # Perform backward pass
        loss.backward()
        # Update model parameters
        optimizer.step()
        # Calculate val_accuracy
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        train_accuracy = 100 * correct / total
        # Log to wandb every $log_interval batches
        if batch_idx % log_interval == 0:
            wandb.log({"epoch":epoch,
                       "batch":batch_idx,
                       "loss": loss.item(),
                       "train_accuracy": train_accuracy})
    # exp lr scheduler step
    if(lrs!="NONE"):
        scheduler.step()
        curr_lr = scheduler.get_last_lr()

    # ============= EVAL ROUTINE ==============
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track performance metrics
    total_correct = 0
    total_samples = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, labeled_batch in enumerate(val_dataloader):
            if (dataset=="iWildCam" or dataset=="rxrx1" or dataset=="fmow"):
                data, targets, metadata = labeled_batch
            else: 
                data, targets = labeled_batch
            # Move data to the appropriate device
            data, targets = data.to(device), targets.to(device)
            # Forward pass
            outputs = model(data)
            val_loss = loss_function(outputs, targets)
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            # Update performance metrics
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()

    # Calculate val_accuracy
    val_accuracy = total_correct / total_samples
    
    # Log val_accuracy to wandb
    wandb.log({"epoch": epoch, "val_accuracy": val_accuracy, "val_loss":val_loss})

    # ========== Save Model =============
    if (val_accuracy > max_acc-acc_threshold and val_loss < min_loss):
        torch.save(model.state_dict(), model_path)
        print(f"Best Model Updated (val_accuracy = {val_accuracy})")
        max_acc = val_accuracy
        min_loss = val_loss
        
    # Print training progress
    print(f"Epoch {epoch+1}/{num_epochs} Completed | Loss: {loss.item():.4f} | Accuracy: {train_accuracy:.2f}% | Val Accuracy: {val_accuracy:.2f} | LR: {curr_lr}")
    model.train()