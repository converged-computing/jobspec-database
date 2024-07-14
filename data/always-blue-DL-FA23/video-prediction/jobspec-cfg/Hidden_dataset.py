import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torch.optim as optim
import datetime
import json
import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torchmetrics

import matplotlib.pyplot as plt
from torchvision import transforms

#local imports
# import model_errors
from nn_models import VideoDataset, HiddenVideoDataset, SimVP, SimVPTAU, MyModel, MyModelSIMVP
from unet_models import ImageDataset, ImageDatasettrainunet, UNet

# read config
with open('config.json', 'r') as file:
    configs = json.load(file)
# print(configs['vp_epochs'])
# print(configs['unet_epochs'])


def datetime_formatted():
    # Get current date and time
    now = datetime.datetime.now()
    # Format the datetime as a string in the specified forma
    formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")
    return str(formatted_now)

logging
logname = '../outs/logs/vp_'+str(datetime_formatted())+'.log'
logging.basicConfig(filename=logname, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

stime = datetime_formatted()
# logging.info("Logging beginning at "+str(stime))
print("Logging beginning at "+str(stime))

transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other transformations here
])
hidden_path = '/scratch/dnp9357'
base_path = '/scratch/dnp9357/dataset'


train_dataset = VideoDataset(base_path, dataset_type='train', transform=transform)
val_dataset = VideoDataset(base_path, dataset_type='val', transform=transform)
unlabeled_dataset = VideoDataset(base_path, dataset_type='unlabeled', transform=transform)
hidden_dataset = HiddenVideoDataset(hidden_path,dataset_type='hidden', transform=transform)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
unlabeled_loader = DataLoader(unlabeled_dataset,batch_size=16,shuffle=True)
hidden_loader = DataLoader(hidden_dataset, batch_size=1, shuffle=False)

# select cuda device if possible
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logging.info(f"Using device: {device}")

# Initialize the model

print("Initialized the modal")
logging.info("This is an info message")

model_save_path = '../outs/models/epoch=42-step=172.ckpt'
#files.download(model_save_path)
shape_in = (11, 3, 160, 240)
print(f"Using device: {device}")
model = MyModel.load_from_checkpoint(model_save_path, shape_in=shape_in).to(device)

# # Load the state dictionary
# state_dict = torch.load(model_save_path)

# # Load the state dict into the model
# model.load_state_dict(state_dict)

print("model loaded")
logging.info("Sim vp model loaded")

batch = next(iter(val_loader))
input_frames, _ = batch
input_frames = input_frames.to(device)

# Predict the 22nd frame
model.eval()
with torch.no_grad():
    predicted_frames = model(input_frames[:, :11])  # Use first 11 frames as input
    predicted_22nd_frame = predicted_frames[:, -1]  # Extract the 22nd frame prediction

# Move tensors to CPU for plotting
predicted_22nd_frame = predicted_22nd_frame.cpu()
actual_22nd_frame = input_frames[:, 21].cpu()  # Actual 22nd frame

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0)  # Change dimensions from CxHxW to HxWxC
    tensor = tensor.numpy()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
    return tensor

# Convert tensors to images
predicted_image = tensor_to_image(predicted_22nd_frame[0])  # First sample in the batch
actual_image = tensor_to_image(actual_22nd_frame[0])  # First sample in the batch

# Plot the images for comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(predicted_image)
plt.title('Predicted 22nd Frame')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(actual_image)
plt.title('Actual 22nd Frame')
plt.axis('off')

# plt.show()
plt.savefig('../outs/images/diff_plot_'+datetime_formatted()+'.png') 


train_dataset_image = ImageDatasettrainunet(base_path, dataset_type='train', transform=transform)
val_dataset_image = ImageDataset(base_path, dataset_type='val', transform=transform)

# Create DataLoaders for each dataset
train_loader_image = DataLoader(train_dataset_image, batch_size=32, shuffle=True)
val_loader_image = DataLoader(val_dataset_image, batch_size=32, shuffle=False)


# ------------------------------------------------ #

# Initialize the U-Net model
modelunet_save_path = '../outs/models/unet_model_2023-12-10_04:07:22.pth'
n_channels = 3  # Assuming RGB images
n_classes = 49  # Update this based on your number of classes
modelunet2 = UNet(n_channels, n_classes).to(device)
print(f"Using device: {device}")
# Load the state dictionary
state_dict_unet = torch.load(modelunet_save_path)

# Load the state dict into the model
modelunet2.load_state_dict(state_dict_unet)
print("model loaded")
logging.info("unet model loaded")
# Assuming 'model' is your trained U-Net model
modelunet2.eval()

    
total_images = 1000  # Total number of images
batch_size = 32      # Batch size from DataLoader
image_height, image_width = 160, 240  # Dimensions of the mask

# Tensor to store ground truth masks
ground_truth_masks = torch.zeros(total_images, image_height, image_width, dtype=torch.long)
  
# Assuming 'model' is your trained U-Net model
modelunet2.eval()

# Total number of images and mask dimensions
total_images = 1000
image_height, image_width = 160, 240

# Tensor to store predictions
predicted_masks = torch.zeros(total_images, image_height, image_width, dtype=torch.long).to(device)

with torch.no_grad():
    for i, (images, masks) in enumerate(val_loader_image):
        # Calculate the start index for this batch
        start_idx_masks = i * batch_size

        # Calculate the end index for this batch
        end_idx_masks = start_idx_masks + masks.shape[0]

        # Store the ground truth masks for this batch
        ground_truth_masks[start_idx_masks:end_idx_masks] = masks
        # Predict the masks for the batch
        images = images.permute(0, 2, 3, 1).to(device)  # Ensure correct shape [batch_size, channels, height, width]
        output = modelunet2(images)
        predicted_batch_masks = torch.argmax(output, dim=1)  # Convert to class indices

        # Calculate the start index for this batch
        start_idx = i * images.shape[0]  # images.shape[0] is the batch size

        # Calculate the end index for this batch
        end_idx = start_idx + images.shape[0]

        # Store the predictions in the tensor
        predicted_masks[start_idx:end_idx] = predicted_batch_masks

# Verify the shape of the ground truth masks tensor
print(ground_truth_masks.shape)  # Should be [1000, 160, 240]

# Verify the shape of the predicted masks tensor
print(predicted_masks.shape)

jaccard = torchmetrics.JaccardIndex(num_classes=49, task="multiclass").to(device)
iou_score = jaccard(predicted_masks, ground_truth_masks.to(device))

print(f"Jaccard Index for INIITAL U-net(IoU) on validation: {iou_score}")
logging.info(f"Jaccard Index for INIITAL U-net(IoU) on validation: {iou_score}")

# ---------------------------- #

model.eval()
model.to(device)
modelunet2.eval()

# Total number of videos and mask dimensions
total_videos = 1000
frame_height, frame_width = 160, 240

# Tensor to store predicted masks
predicted_masks_simvp = torch.zeros(total_videos, frame_height, frame_width, dtype=torch.long).to(device)
groundtruth_masks_simvp = torch.zeros(total_videos, frame_height, frame_width, dtype=torch.long).to(device)

with torch.no_grad():
    for i, (videos, masks) in enumerate(val_loader):
        # videos shape is expected to be [1, frames, channels, height, width]
        # Extract the first 11 frames
        predicted_frames = model(videos[:, :11].to(device))  # Use first 11 frames as input
        predicted_22nd_frame = predicted_frames[:, -1]  # Extract the 22nd frame prediction

        # Reshape or process predicted_22nd_frame as required by model_unet
        # Assuming model_unet expects [batch_size, channels, height, width]
        predicted_22nd_frame = predicted_22nd_frame.permute(0, 1,2, 3).to(device)

        # Predict the semantic mask of the predicted 22nd frame
        mask_output = modelunet2(predicted_22nd_frame)
        predicted_mask = torch.argmax(mask_output, dim=1).squeeze(0)

        # Store the predicted mask
        groundtruth_masks_simvp[i] = masks[:, -1, :, :].squeeze(0).to(device)
        predicted_masks_simvp[i] = predicted_mask

# Verify the shape of the predicted masks tensor
print(predicted_masks_simvp.shape)  # Should be [1000, 160, 240]
print(groundtruth_masks_simvp.shape)
# ---------------------------- #

jaccard = torchmetrics.JaccardIndex(num_classes=49, task="multiclass").to(device)
iou_score = jaccard(predicted_masks_simvp, groundtruth_masks_simvp.to(device))

print(f"Jaccard Index (IoU) for LINKED model on validation: {iou_score}")
logging.info(f"Jaccard Index (IoU) for LINKED model on validation: {iou_score}")


model.eval()
model.to(device)
modelunet2.eval()

# Total number of videos and mask dimensions
total_videos = 2000
frame_height, frame_width = 160, 240

# Tensor to store predicted masks
predicted_masks_simvp_hidden = torch.zeros(total_videos, frame_height, frame_width, dtype=torch.long).to(device)
#groundtruth_masks_simvp = torch.zeros(total_videos, frame_height, frame_width, dtype=torch.long).to(device)

with torch.no_grad():
    for i, (videos) in enumerate(hidden_loader):
        # videos shape is expected to be [1, frames, channels, height, width]
        # Extract the first 11 frames
        predicted_frames = model(videos[:, :11].to(device))  # Use first 11 frames as input
        predicted_22nd_frame = predicted_frames[:, -1]  # Extract the 22nd frame prediction

        # Reshape or process predicted_22nd_frame as required by model_unet
        # Assuming model_unet expects [batch_size, channels, height, width]
        predicted_22nd_frame = predicted_22nd_frame.permute(0, 1,2, 3).to(device)

        # Predict the semantic mask of the predicted 22nd frame
        mask_output = modelunet2(predicted_22nd_frame)
        predicted_mask = torch.argmax(mask_output, dim=1).squeeze(0)

        # Store the predicted mask
        predicted_masks_simvp_hidden[i] = predicted_mask

# Verify the shape of the predicted masks tensor
print(predicted_masks_simvp_hidden.shape)  # Should be [1000, 160, 240]
logging.info(f"Shape of hidden tensor is: {predicted_masks_simvp_hidden.shape}")
# Convert to NumPy array
teamid30_alwaysblue = predicted_masks_simvp_hidden.cpu().numpy()  # Use tensor.cpu().numpy() if your tensor is on GPU

# Save as .npy file
np.save('teamid30_alwaysblue.npy', teamid30_alwaysblue) 

