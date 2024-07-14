from transformers import SamModel, SamProcessor
import torch
from dataset import S1S2Dataset
import numpy as np
import rasterio
import os
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pandas as pd

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)  # Transfert sur CPU avec .cpu()
    ax.imshow(mask_image)


def compute_metrics(preds, labels):
    metrics = {}
    # IOU
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    metrics["iou"] = iou.mean().item()
    
    # Dice
    dice = (2.0 * intersection + 1e-6) / (preds.float().sum((1, 2)) + labels.float().sum((1, 2)) + 1e-6)
    metrics["dice"] = dice.mean().item()
    
    # Recall
    recall = intersection / (labels.float().sum((1, 2)) + 1e-6)
    metrics["recall"] = recall.mean().item()
    
    # Precision
    precision = intersection / (preds.float().sum((1, 2)) + 1e-6)
    metrics["precision"] = precision.mean().item()
    
    return metrics

def main(checkpoint_path, split_path, ndwi):
    # save dir 
    # prendre la date dans le checkpoint c'est l'avant dernier avant /
    date = checkpoint_path.split("/")[-2]
    folder = "eval_" + date + "_ndwi_" + str(ndwi)
    save_dir = Path("results") / folder
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    
    metrics = []
    
    # device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load dataset 
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataset = S1S2Dataset(os.path.join(split_path,"train/img"), os.path.join(split_path,"train/msk"), processor, ndwi=ndwi)
    val_loader = DataLoader(dataset, batch_size=8 , shuffle=False)
    
    # load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("load model")
    model.eval()
    
    csv_file = os.path.join(save_dir, "metrics.csv")
    header_written = False
    for i, data in enumerate(val_loader):
        inputs, labels = data
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs["pixel_values"], multimask_output=False)
            preds = torch.sigmoid(outputs.pred_masks.squeeze(1))
            preds = (preds > 0.5).to(torch.uint8)
            preds = preds.to(torch.uint8)
            labels = labels.to(torch.uint8)
            # Convertir en type uint8
            metric = compute_metrics(preds, labels)
            metrics.append(metric)
            
            raw = {'batch':i, 'iou': metric["iou"], 'dice': metric["dice"], 'recall': metric["recall"], 'precision': metric["precision"]}
            M = pd.DataFrame(raw, index=[0])
            # Écriture dans le fichier CSV
            if not header_written:
                M.to_csv(csv_file, index=False)  # Écriture de l'en-tête
                header_written = True  # Mettre à jour la variable pour indiquer que l'en-tête a été écrit
            else:
                M.to_csv(csv_file, mode='a', header=False, index=False) 
            

            # Enregistrer les prédictions et les masques réels
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            img = np.array(inputs["pixel_values"][0].cpu().numpy().transpose(1, 2, 0))
            axes[0].imshow(np.array(inputs["pixel_values"][0].cpu().numpy().transpose(1, 2, 0)))
            show_mask(preds[0], axes[0])
            axes[0].set_title("Predicted mask")
            axes[0].axis("off")
            axes[1].imshow(np.array(inputs["pixel_values"][0].cpu().numpy().transpose(1, 2, 0)))
            show_mask(labels[0], axes[1])
            axes[1].set_title("Ground truth mask")
            axes[1].axis("off")
            plt.savefig(os.path.join(save_dir, f'figure_{i}.png'))
            plt.close()
 
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="SAM-fine-tune Inference")
    parser.add_argument("--checkpoint_path", required = True, help="checkpoint.")
    parser.add_argument("--split_path", required = True, help="The file to perform inference on.")
    parser.add_argument("--ndwi", required = True, help="Using RGB or R-NDWI-B")
    
    args = parser.parse_args()
    split_path = args.split_path
    ndwi = args.ndwi
    checkpoint_path = args.checkpoint_path
    
    main(checkpoint_path,split_path, ndwi)