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
seed,batch_size,epochs = 64,8,200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ID = sys.argv[1]
TASK = "v2lr"
CONDUCTANCE_VALUES = ""
DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"
recon_path = "./models/img/14.2.1.retraining.2.20231130014311_img.pt" # "./models/img/14.2.1.20231116190651_img.pt" #"./models/img/14.2.20231110000321.pt"

diff_transform = transforms.Compose([transforms.ToTensor()])
img_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = DiffImg(csv_file=DIFFS_IMGS_TRAIN_PATH, diff_transform=diff_transform, img_transform=img_transform,npf=False,ppf=False,zero_background=True)
test_dataset = DiffImg(csv_file=DIFFS_IMGS_TEST_PATH, diff_transform=diff_transform, img_transform=img_transform,npf=False,ppf=False,zero_background=True)

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

def stretch_diff(batch_diff):
    batch_capsule = torch.zeros((batch_diff.shape[0],batch_diff.shape[1],batch_diff.shape[2]*2,batch_diff.shape[3]*2))
    for i,diff in enumerate(batch_diff): # batch_diff.shape = (32,1,16,16)
        diff = diff.squeeze()
        mask = torch.triu(torch.ones_like(diff), diagonal=1)

        # Apply the mask to select the upper triangle
        upper_triangle = mask*3
        lower_triangle = (1 - mask)

        capsule = torch.zeros((diff.shape[0]*2,diff.shape[1]*2))
        capsule[:16,:16] = diff
        capsule[:16,16:] = lower_triangle
        capsule[16:,:16] = upper_triangle

        batch_capsule[i] = capsule
    
    return batch_capsule

def adjusted_mse(batch_img, batch_recon):
    mask = (torch.Tensor(batch_img) != 0.0).float() # check if batch_img is translated to 0~
    num_valid_pixels = torch.sum(mask)
    squared_diff = (batch_img - batch_recon).to(device)**2 * mask.to(device)
    mse = torch.sum(squared_diff) / (num_valid_pixels)

    return mse 

criterion = VGGPerceptualLoss().to(device)

def play(dataloader:DataLoader=None,
         tracker:LossTracker=None,
         v2lr:nn.Module=None,
         ssim=None,optimizer=None,
         lossid=0):
    
    v2lr.train() if tracker.job=="Training" else v2lr.eval()
    if tracker.best_epoch == -1 and tracker.job == "Training":
        print("Ready to TRAIN!!")
    for i, (batch_diff, batch_img) in enumerate(dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        batch_mapped, batch_lr, batch_recon_v = v2lr(batch_diff,batch_img)

        # scale_to_input removed from here
        mse_loss = F.mse_loss(batch_img, batch_recon_v)
        ssim_value = 1 - ssim(batch_img, batch_recon_v) 
        vgg_loss = criterion(batch_img,batch_recon_v)
        mse_loss_lr = F.mse_loss(batch_lr, batch_mapped)
        
        # 0 MSE
        if lossid==0:
            loss = mse_loss
        # 1 SSIM
        elif lossid==1:
            loss=ssim_value
        # 2 alpha*mse + (1-alpha)*ssim
        elif lossid==2:
            alpha=0.05
            loss = alpha*mse_loss + (1-alpha)*ssim_value
        # 3 alpha*mse + (1-alpha)*0.1*ssim
        elif lossid==3:
            alpha=0.1
            loss = alpha*mse_loss + (1-alpha)*0.1*ssim_value
        # 4 mse/ssim+1
        elif lossid==4:
            loss=mse_loss/(ssim_value+1)
        # 5 mse_aoi
        elif lossid==5:
            loss = adjusted_mse(batch_img,batch_recon_v)
        # 6 vgg_loss
        elif lossid==6:
            loss=vgg_loss
        # 7 vgg_loss+10*mse_aoi
        elif lossid==7:
            loss=vgg_loss+10*adjusted_mse(batch_img,batch_recon_v) 

        if tracker.job == "Training": # batch
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()


        tracker.epoch_mse_loss += mse_loss.item()
        tracker.epoch_ssim += ssim_value.item()
        tracker.epoch_loss += loss.item()
        tracker.epoch_loss_lr += mse_loss_lr.item()
        tracker.epoch_vgg_loss += vgg_loss.item()

    tracker.epoch_mse_loss /= (i+1)
    tracker.epoch_ssim /= (i+1)
    tracker.epoch_loss /= (i+1)
    tracker.epoch_loss_lr /= (i+1)
    tracker.epoch_vgg_loss /= (i+1)

    # epoch rebackprog removed from here
    return tracker

if ID=="0f.3":
    v2l = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256,216),
        # nn.ReLU(),
    )
    v2lr_path = "./models/v2lr/0f.1.20231226165034_v2lr.pt"

elif ID=="1f.3":
    v2l = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128,216),
        nn.ReLU()
    )
    v2lr_path = "./models/v2lr/1f.1.20231226165408_v2lr.pt"

elif ID=="1.1f.3":
    v2l = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Linear(128,216),
        nn.ReLU()
    )
    v2lr_path = "./models/v2lr/1.1f.1.20231226165615_v2lr.pt"

elif ID=="1.2f.3":
    v2l = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256, 324),
        nn.ReLU(),
        nn.Linear(324,256),
        nn.ReLU(),
        nn.Linear(256,324),
        nn.ReLU(),
        nn.Linear(324,216),
        nn.ReLU()
    )
    v2lr_path = "./models/v2lr/1.2f.1.20231226184514_v2lr.pt"

elif ID=="2f.3":
    v2l = nn.Sequential(
        nn.Conv2d(1,6,5,1,0),
        nn.Conv2d(6,6,3,1,1),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        
        nn.Conv2d(6,18,5,1,0),
        nn.Conv2d(18,18,3,1,1),
        nn.BatchNorm2d(18),
        nn.ReLU(),
        
        nn.Conv2d(18,54,5,1,0),
        nn.Conv2d(54,54,3,1,1),
        nn.BatchNorm2d(54),
        nn.ReLU(),
        
        nn.Conv2d(54,108,3,1,0),
        nn.Conv2d(108,108,3,1,1),
        nn.BatchNorm2d(108),
        nn.ReLU(),   
        
        nn.Conv2d(108,216,2,1,0),
        nn.Conv2d(216,216,3,1,1),
        nn.BatchNorm2d(216),
        nn.ReLU(),
        
        nn.Flatten()
    )
    v2lr_path = "./models/v2lr/2f.1.20231226191152_v2lr.pt"

elif ID=="2.1f.3":
    v2l = nn.Sequential(
            ResidualBlock(1,6,5,1,0),
            nn.BatchNorm2d(6),
            ResidualBlock(6,18,5,1,0),
            nn.BatchNorm2d(18),
            ResidualBlock(18,54,5,1,0),
            nn.BatchNorm2d(54),
            ResidualBlock(54,108,3,1,0),
            nn.BatchNorm2d(108),   
            ResidualBlock(108,216,2,1,0),
            nn.BatchNorm2d(216),
            nn.Flatten()
        )
    v2lr_path = "./models/v2lr/2.1f.1.20231226195929_v2lr.pt"

# idk=[-1]
for i in [7]:
    # if i ==0:
    #     continue
    print(f"LossID: {i}")
    nownow = datetime.now()
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"
    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

    # v2lr = V2ImgLR(recon_path,train_recon=True)
    # v2lr.v2lr = config
    # v2lr = torch.load("./models/v2lr/0.alpha.01.20231206000636_v2lr.pt")
    # v2lr = torch.load("./models/v2lr/1.5.vgg.ppf.20231223191000_v2lr.pt",map_location=device)
    v2lr = torch.load(v2lr_path,map_location=device)
    # v2lr.v2lr = v2l
    v2lr = v2lr.to(device)
    for d in v2lr.recon.decoder.parameters():
        d.requires_grad = False

    for v in v2lr.v2lr.parameters():
        v.requires_grad = True
    summary(v2lr.v2lr,(1,16,16))
    # print(v2lr.recon.decoder)
    summary(v2lr.recon.decoder,(216))
    # summary(v2lr,(1,16,16))
    
    ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
    
    optimizer_config = {'Adam': {'learning_rate':1e-3, 'weight_decay':0}}    
    optimizer = optimizer_build(optimizer_config,v2lr)

    best_v2lr = deepcopy(v2lr)
    min_loss = np.inf
    best_epoch = 0

    train_losses = []
    last_printed = 0
    tolerance = 3
    for epoch in range(epochs):
        trainer = play(train_dataloader,trainer,v2lr,ssim,optimizer,lossid=i)
        train_losses.append(trainer.epoch_mse_loss)
        
        with torch.no_grad():
            validator = play(val_dataloader,validator,v2lr,ssim,lossid=i)

        if validator.epoch_mse_loss < min_loss:
            min_loss = validator.epoch_mse_loss
            
            del best_v2lr
            best_v2lr = deepcopy(v2lr)
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
            print(f"Tolerance: {tolerance}!! {trainer} !==! {validator}")
            tolerance -= 1
            
        if tolerance == 0:
            break

    del v2lr
    v2lr = deepcopy(best_v2lr)
    with torch.no_grad():
        tester = play(test_dataloader,tester,v2lr,ssim,lossid=i)
        
        train_losses.append(tester.epoch_loss)
        print(tester)

    torch.save(v2lr, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    with open(LOSS_TRACKER_PATH, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_losses)

    print(f"written to: {LOSS_TRACKER_PATH}")
    end = time.time()
    print(f"Elapsed time: {end - start} seconds.")
    start = end