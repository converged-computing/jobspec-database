
import numpy as np
import rasterio
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset,  DataLoader
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import torch.nn.functional as F
import numpy as np
import tqdm
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from torchsummary import summary
from torch.optim import Adam
import csv
import pandas as pd
import argparse

from dataset import S1S2Dataset
def plot_loss_curve(loss_values, file_name):
    plt.figure()
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.savefig(file_name)
    plt.close()


def main(split_path, ndwi):
    # dataset 
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataset = S1S2Dataset(os.path.join(split_path,"train/img"), os.path.join(split_path,"train/msk"), processor, ndwi=ndwi)
    train_loader = DataLoader(dataset, batch_size=8 , shuffle=True)
    
    # Fine-tuning mask decoder 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
            
    # save dir 
    save_dir = Path("results") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # create dir 
    os.makedirs(save_dir, exist_ok=True)
    
            
    loss_values = []
    
    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-4, weight_decay=0)

    model.train()
    
    csv_file = os.path.join(save_dir, 'loss_per_epoch.csv')
    header_written = False
    for epoch in range(24):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.float().to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(pixel_values=inputs["pixel_values"], multimask_output=False)
            
            predicted_masks = outputs.pred_masks.squeeze(1)
            loss = criterion(predicted_masks, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        losses = {'epoch':epoch + 1,'loss': running_loss/ len(train_loader)}
        L = pd.DataFrame(losses, index=[0])
        # Écriture dans le fichier CSV
        if not header_written:
            L.to_csv(csv_file, index=False) 
            header_written = True  
        else:
            L.to_csv(csv_file, mode='a', header=False, index=False) 
            

        if (epoch + 1) % 3 == 0:
            checkpoint_filename = os.path.join(save_dir, 'checkpoint_{}.pt'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                }, checkpoint_filename)
            print('Checkpoint saved at epoch {} to {}'.format(epoch+1, checkpoint_filename))

    # Plot and save loss curve
    plot_loss_curve(loss_values,file_name= os.path.join(save_dir, 'loss_curve.png'))

    print('Finished Training')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-fine-tune training")
    parser.add_argument("--split_path", required = True, help="The file to perform inference on.")
    parser.add_argument("--ndwi", required = True, help="Using RGB or R-NDWI-B")
    
    args = parser.parse_args()
    split_path = args.split_path
    ndwi = args.ndwi
    
    main(split_path, ndwi)
        