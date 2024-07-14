import sys
import time
import csv
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.classes import *

print("Importing finished!!")

start = time.time()
seed,batch_size,epochs = 64,4,200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ID = sys.argv[1]
TASK = "img"
CONDUCTANCE_VALUES = ""
DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"
recon_path = "./models/img/14.2.2.20231210034221_img.pt"# "./models/img/14.2.1.retraining.2.20231130014311_img.pt" # "./models/img/14.2.1.20231116190651_img.pt" #"./models/img/14.2.20231110000321.pt"

diff_transform = transforms.Compose([transforms.ToTensor()])
img_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = DiffImg(csv_file=DIFFS_IMGS_TRAIN_PATH, diff_transform=diff_transform, img_transform=img_transform)
test_dataset = DiffImg(csv_file=DIFFS_IMGS_TEST_PATH, diff_transform=diff_transform, img_transform=img_transform)

generator = torch.Generator().manual_seed(seed)
train_size = int(0.8 * len(train_dataset)) 
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Dataset loaded!! Length (train dataset) - {len(train_dataset)}")

trainer = LossTracker("Training")
validator = LossTracker("Validation")
tester = LossTracker("Testing")

def compute_mse_aoi(recons,preds):
    mses_aoi = torch.tensor(0.0, device=device)
    for recon,pred in zip(recons,preds):
        vals,counts = recon.unique(return_counts=True)

        foreground_value = vals[counts.argmin()]
        aoi = (recon == foreground_value).nonzero()
        xmin=aoi[:,1].min()
        xmax=aoi[:,1].max()
        ymin=aoi[:,2].min()
        ymax=aoi[:,2].max()
        recon_aoi = recon[:, xmin:xmax + 1, ymin:ymax + 1]
        pred_aoi = pred[:, xmin:xmax + 1, ymin:ymax + 1]

        mse_aoi=F.mse_loss(recon_aoi, pred_aoi)
        mses_aoi += mse_aoi

    mse_aoi = mses_aoi/len(recons)
    return mse_aoi

def play(dataloader:DataLoader=None,
         tracker:LossTracker=None,
         recon:nn.Module=None,
         ssim=None,optimizer=None):    
    
    recon.train()  if tracker.job=="Training" else recon.eval()
    
    for i, (_, batch_img) in enumerate(dataloader):
        batch_img = batch_img.to(device)
        _,batch_decoded = recon(batch_img)
        
        # scale_to_input removed from here
            
        mse_loss = F.mse_loss(batch_img, batch_decoded)
        ssim_value = 1 - ssim(batch_img, batch_decoded) 
        mse_aoi = compute_mse_aoi(batch_img,batch_decoded)
        loss = mse_aoi
     
        if tracker.job == "Training": # batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tracker.epoch_mse_loss += mse_loss.item()
        tracker.epoch_ssim += ssim_value.item()
        tracker.epoch_loss += loss.item()

    tracker.epoch_mse_loss /= len(dataloader)
    tracker.epoch_ssim /= len(dataloader)
    tracker.epoch_loss /= len(dataloader)

    # epoch rebackprog removed from here
    return tracker

configs=[216]
for i,config in enumerate(configs):
    print(f"Config {config}")
    nownow = datetime.now()
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"
    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

    recon = torch.load(recon_path)
    recon = recon.to(device)
    print(recon)

    ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
    
    optimizer_config = {'Adam': {'learning_rate':1e-4,'weight_decay':1e-5}}
    optimizer = optimizer_build(optimizer_config,recon)

    best_recon = deepcopy(recon)
    min_loss = np.inf
    best_epoch = 0

    train_losses = []
    last_printed = 0
    for epoch in range(epochs):
        trainer = play(train_dataloader,trainer,recon,ssim,optimizer)
        train_losses.append(trainer.epoch_mse_loss)
        
        with torch.no_grad():
            validator =play(val_dataloader,validator,recon,ssim)

        if validator.epoch_loss < min_loss :
            min_loss = validator.epoch_loss
            
            del best_recon
            best_recon = deepcopy(recon)
            
            best_epoch = epoch
            trainer.best_epoch = best_epoch
            validator.best_epoch = best_epoch

            print(f"{trainer} !==! {validator}")
            last_printed = epoch
        
        if epoch - last_printed > 20:
            last_printed = epoch

            # note that this is not the best epoch
            trainer.best_epoch = epoch
            validator.best_epoch = epoch
            print(f"Early stopping!! {trainer} !==! {validator}")
            break

    del recon
    recon = deepcopy(best_recon)
    with torch.no_grad():
        tester = play(test_dataloader,tester,recon,ssim)
        
        train_losses.append(tester.epoch_loss)
        print(tester)

    torch.save(recon, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    with open(LOSS_TRACKER_PATH, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_losses)

    print(f"written to: {LOSS_TRACKER_PATH}")
    end = time.time()
    print(f"Elapsed time: {end - start} seconds.")
    start = end