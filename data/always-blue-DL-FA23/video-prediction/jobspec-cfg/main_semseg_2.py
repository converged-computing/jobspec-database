import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR
import torch.optim as optim
import datetime
import json
import logging
import numpy as np

import torchmetrics

import matplotlib.pyplot as plt
from torchvision import transforms

#local imports
# import model_errors
from nn_models import VideoDataset, SimVP, HiddenVideoDataset
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

# logging
logname = '../outs/logs/vp_'+str(datetime_formatted())+'.log'
logging.basicConfig(filename=logname, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

stime = datetime_formatted()
logging.info("Logging beginning at "+str(stime))
print("Logging beginning at "+str(stime))

transform = transforms.Compose([
    # transforms.Resize((height, width)), # Specify your desired height and width
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(), # You can specify parameters like brightness, contrast, etc.
    transforms.ToTensor(),
    # transforms.Normalize(mean, std) # Specify the mean and std for your dataset
])
base_path = '../dataset'
base_path = '/scratch/dnp9357/dataset'

train_dataset = VideoDataset(base_path, dataset_type='train', transform=transform)
val_dataset = VideoDataset(base_path, dataset_type='val', transform=transform)
unlabeled_dataset = VideoDataset(base_path, dataset_type='unlabeled', transform=transform)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset,batch_size=16,shuffle=True)


# #training
# epochs=10
shape_in = (11, 3, 128, 128)  # You need to adjust these dimensions based on your actual data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# logging.info(f"Using device: {device}")

# # Initialize the model
# model = SimVP(shape_in=shape_in).to(device)
# model.train()

# frame_prediction_criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# # OneCycleLR Scheduler
# total_steps = epochs * (len(train_loader)+len(unlabeled_loader)) *2  # Total number of training steps
# scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)

# print("before training")
# logging.info("This is an info message")


# # for epoch in range(int(configs['vp_epochs']):
# for epoch in range(epochs):

#     # first train on unlabeled dataset
#     for batch in unlabeled_loader:
#         images, _ = batch
        
#         input_frames = images[:, :11].to(device)
#         target_frame = images[:, 21].to(device)

#         # Forward pass
#         predicted_frames = model(input_frames)
#         predicted_target_frame = predicted_frames[:, -1]

#         # Loss computation
#         loss = frame_prediction_criterion(predicted_target_frame, target_frame)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Update the learning rate
#         scheduler.step()

#         # print(f"Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
#         logging.info(f"Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")

# for epoch in range(epochs):
#     # now train on training dataset
#     for batch in train_loader:
#         images, _ = batch
#         input_frames = images[:, :11].to(device)
#         target_frame = images[:, 21].to(device)

#         # Forward pass
#         predicted_frames = model(input_frames)
#         predicted_target_frame = predicted_frames[:, -1]

#         # Loss computation
#         loss = frame_prediction_criterion(predicted_target_frame, target_frame)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Update the learning rate
#         scheduler.step()

#         # print(f"Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
#         logging.info(f"Epoch [{epoch+1}/{epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")

# model_save_path = '../outs/models/my_model_' +str(datetime_formatted())+'.pth'

# # Save the model's state dictionary
# torch.save(model.state_dict(), model_save_path)

# # Inform the user
# print(f'Model saved to {model_save_path}')
# logging.info(f'Model saved to {model_save_path}')


# # model_save_path = 'my_model.pth'
# #files.download(model_save_path)
# shape_in = (11, 3, 128, 128)
# print(f"Using device: {device}")
# model = SimVP(shape_in=shape_in).to(device)

# # Load the state dictionary
# state_dict = torch.load(model_save_path)

# # Load the state dict into the model
# model.load_state_dict(state_dict)


# model.eval()
# model.to(device)
# mse_loss = nn.MSELoss()
# total_loss = 0.0
# with torch.no_grad():  # Disable gradient computation
#     for batch in val_loader:
#         images, _ = batch
#         input_frames = images[:, :11].to(device)  # First 11 frames
#         actual_22nd_frame = images[:, 21].to(device)
#         # Forward pass to get the predictions
#         predicted_frames = model(input_frames)
#         predicted_22nd_frame = predicted_frames[:, -1]
#         loss = mse_loss(predicted_22nd_frame, actual_22nd_frame)
#         total_loss += loss.item()

# # Calculate the average loss
# average_loss = total_loss / len(val_loader)
# print(f"Average MSE Loss on the validation dataset: {average_loss}")
# logging.info(f"Average MSE Loss on the validation dataset: {average_loss}")


# batch = next(iter(val_loader))
# input_frames, _ = batch
# input_frames = input_frames.to(device)

# # Predict the 22nd frame
# model.eval()
# with torch.no_grad():
#     predicted_frames = model(input_frames[:, :11])  # Use first 11 frames as input
#     predicted_22nd_frame = predicted_frames[:, -1]  # Extract the 22nd frame prediction

# # Move tensors to CPU for plotting
# predicted_22nd_frame = predicted_22nd_frame.cpu()
# actual_22nd_frame = input_frames[:, 21].cpu()  # Actual 22nd frame

# # Function to convert tensor to image
# def tensor_to_image(tensor):
#     tensor = tensor.squeeze(0)  # Remove batch dimension
#     tensor = tensor.permute(1, 2, 0)  # Change dimensions from CxHxW to HxWxC
#     tensor = tensor.numpy()
#     tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
#     return tensor

# # Convert tensors to images
# predicted_image = tensor_to_image(predicted_22nd_frame[0])  # First sample in the batch
# actual_image = tensor_to_image(actual_22nd_frame[0])  # First sample in the batch

# # Plot the images for comparison
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(predicted_image)
# plt.title('Predicted 22nd Frame')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(actual_image)
# plt.title('Actual 22nd Frame')
# plt.axis('off')

# # plt.show()
# plt.savefig('../outs/images/diff_plot_'+datetime_formatted()+'.png') 



#keep commented
# unet training


# Initialize the U-Net model
# n_channels = 3  # Assuming RGB images
# n_classes = 49  # Update this based on your number of classes
# modelunet = UNet(n_channels, n_classes).to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(modelunet.parameters(), lr=0.001)
# # Learning rate scheduler
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Number of epochs
# num_epochs = config.unet_epochs
# num_epochs = 10
# total_steps = num_epochs * len(train_loader)

# for epoch in range(num_epochs):
#     modelunet.train()
#     running_loss = 0.0

#     # Training loop
#     for images, masks in train_loader:
#         # Assuming the input images are of shape (batch_size, frames, C, H, W)
#         inputs = images[:, -1, :, :, :].to(device)
#         #print(torch.unique(masks[0,0,:,:]))
#         masks = masks[:, -1, :, :].to(device)  # Adjust dimensions if necessary
#         #masks = torch.argmax(masks, dim=1)
#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward + backward + optimize
#         outputs = modelunet(inputs)
#         print(f'Inputs Shape: {inputs.shape}')
#         print(f'Masks Shape: {masks.shape}')
#         print(f'Outputs Shape: {outputs.shape}')
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")

#     # Print training statistics
#     train_loss = running_loss / len(train_loader)
#     print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')


train_dataset_image = ImageDatasettrainunet(base_path, dataset_type='train', transform=transform)
val_dataset_image = ImageDataset(base_path, dataset_type='val', transform=transform)

# Create DataLoaders for each dataset
train_loader_image = DataLoader(train_dataset_image, batch_size=32, shuffle=True)
val_loader_image = DataLoader(val_dataset_image, batch_size=32, shuffle=True)


# ------------------------------------------------ #

# Initialize the U-Net model
n_channels = 3  # Assuming RGB images
n_classes = 49  # Update this based on your number of classes
modelunet2 = UNet(n_channels, n_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelunet2.parameters(), lr=0.01)
# Learning rate scheduler
# scheduler = StepLR(optimizer, step_size=3500, gamma=0.1, verbose=False)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Number of epochs
num_epochs = 200
total_steps = num_epochs * len(train_loader_image)

unet_model_save_path = '../outs/models/unet_model_'+datetime_formatted()+'.pth'

model_path = '../outs/models/unet_model_2023-12-10_04:07:22.pth'
# Check if the model file exists
if os.path.exists(model_path):
    # Load the model
    modelunet2.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
    logging.info(f"Loaded model from {model_path}")
    unet_model_save_path = model_path
else:
    print(f"No pre-trained model found at {model_path}")
    logging.info(f"Loaded model from {model_path}")



for epoch in range(num_epochs):
    modelunet2.train()
    running_loss = 0.0

    # Training loop
    for images, masks in train_loader_image:
        # Assuming the input images are of shape (batch_size, frames, C, H, W)
        #images = images.permute(0, 2, 1, 3)
        inputs = images.permute(0,2,3,1).to(device)
        #print(torch.unique(masks[0,0,:,:]))
        masks = masks.to(device)  # Adjust dimensions if necessary
        #masks = torch.argmax(masks, dim=1)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = modelunet2(inputs)
        #print(f'Inputs Shape: {inputs.shape}')
        #print(f'Masks Shape: {masks.shape}')
        #print(f'Outputs Shape: {outputs.shape}')
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        # Update the learning rate
        # scheduler.step(loss)
        running_loss += loss.item()
        # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{scheduler.last_epoch}/{total_steps}], Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
    # Print training statistics
    train_loss = running_loss / len(train_loader_image)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')

    if (epoch + 1) % 20 == 0:
        torch.save(modelunet2.state_dict(), unet_model_save_path)




# Save the model's state dictionary
torch.save(modelunet2.state_dict(), unet_model_save_path)

# ------------------------------------------------ #

# Function to generate a color palette
def generate_palette(num_classes):
    np.random.seed(42)  # For reproducible colors
    palette = np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)
    return palette

# Function to convert mask indices to a color image
def mask_to_color(mask, palette):
    # Create an RGB image from the mask
    color_mask = palette[mask]
    return color_mask
    

# Generate a color palette for 49 classes
palette = generate_palette(49)

# ------------------------------------------------ #
# this section for demo purposes

# Assuming 'model' is your trained U-Net model
# modelunet2.eval()
# 
# with torch.no_grad():
#     # Iterate through the validation dataset
#     for i, (images, masks) in enumerate(val_loader_image):
#         # Process one sample for demonstration purposes
#         if i == 0:  # Adjust the index if you want to display a different sample
#             # Ensure images are in the shape [batch_size, channels, height, width]
#             images = images.permute(0, 2, 3, 1).to(device)

#             # Predict the mask
#             outputs = modelunet2(images)
#             predicted_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()
#             # predicted_color_mask = mask_to_color(predicted_mask.cpu().numpy(), palette)

#             # Actual mask
#             actual_mask = masks[0].cpu().numpy()
#             # actual_color_mask = mask_to_color(actual_mask, palette)

#             # Plotting
#             fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#             axs[0].imshow(predicted_mask)
#             axs[0].set_title('Predicted Mask')
#             axs[0].axis('off')

#             axs[1].imshow(actual_mask)
#             axs[1].set_title('Actual Mask')
#             axs[1].axis('off')

#             plt.show()
#             break

# ------------------------------------------------ #

total_images = 1000  # Total number of images
batch_size = 32      # Batch size from DataLoader
image_height, image_width = 160, 240  # Dimensions of the mask

# Tensor to store ground truth masks
ground_truth_masks = torch.zeros(total_images, image_height, image_width, dtype=torch.long)

with torch.no_grad():
    for i, (_, masks) in enumerate(val_loader_image):
        # Calculate the start index for this batch
        start_idx = i * batch_size

        # Calculate the end index for this batch
        end_idx = start_idx + masks.shape[0]

        # Store the ground truth masks for this batch
        ground_truth_masks[start_idx:end_idx] = masks

# Verify the shape of the ground truth masks tensor
print(ground_truth_masks.shape)  # Should be [1000, 160, 240]

# ------------------------------------------------ #


# Assuming 'model' is your trained U-Net model
modelunet2.eval()

# Total number of images and mask dimensions
total_images = 1000
image_height, image_width = 160, 240

# Tensor to store predictions
predicted_masks = torch.zeros(total_images, image_height, image_width, dtype=torch.long).to(device)

with torch.no_grad():
    for i, (images, _) in enumerate(val_loader_image):
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

with torch.no_grad():
    for i, (videos, _) in enumerate(val_loader):
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
        predicted_masks_simvp[i] = predicted_mask

# Verify the shape of the predicted masks tensor
print(predicted_masks_simvp.shape)  # Should be [1000, 160, 240]

# ---------------------------- #

jaccard = torchmetrics.JaccardIndex(num_classes=49, task="multiclass").to(device)
iou_score = jaccard(predicted_masks_simvp, ground_truth_masks.to(device))

print(f"Jaccard Index (IoU) for LINKED model on validation: {iou_score}")
logging.info(f"Jaccard Index (IoU) for LINKED model on validation: {iou_score}")

# ---------------------------- #

model.eval()
model.to(device)
modelunet2.eval()

# Total number of videos and mask dimensions
total_videos = 1000
frame_height, frame_width = 160, 240
batch = next(iter(val_loader))
input_frames, maskstest = batch
input_frames = input_frames.to(device)

predicted_frames = model(input_frames[:, :11])  # Use first 11 frames as input
predicted_22nd_frame = predicted_frames[:, -1]  # Extract the 22nd frame prediction
predicted_22nd_frame = predicted_22nd_frame.permute(0, 1,2, 3).to(device)
 # Predict the mask
outputs = modelunet2(predicted_22nd_frame)
predicted_mask = torch.argmax(outputs, dim=1)[0]
predicted_color_mask = mask_to_color(predicted_mask.cpu().numpy(), palette)

# Actual mask
actual_mask = maskstest[:, -1, :, :].squeeze(0).cpu().numpy()
actual_color_mask = mask_to_color(actual_mask, palette)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(predicted_color_mask)
axs[0].set_title('Predicted Mask')
axs[0].axis('off')

axs[1].imshow(actual_color_mask)
axs[1].set_title('Actual Mask')
axs[1].axis('off')

plt.show()
plt.savefig('../outs/images/final_out_'+str(datetime_formatted()))



# ---------------------------- #


# model.eval()
# model.to(device)
# modelunet2.eval()

# # Total number of videos and mask dimensions
# total_videos = 2000
# frame_height, frame_width = 160, 240

# # Tensor to store predicted masks
# predicted_masks_simvp = torch.zeros(total_videos, frame_height, frame_width, dtype=torch.long).to(device)

# with torch.no_grad():
#     for i, (videos, _) in enumerate(hidden_loader):
#         # videos shape is expected to be [1, frames, channels, height, width]
#         # Extract the first 11 frames
#         predicted_frames = model(videos[:, :11].to(device))  # Use first 11 frames as input
#         predicted_22nd_frame = predicted_frames[:, -1]  # Extract the 22nd frame prediction

#         # Reshape or process predicted_22nd_frame as required by model_unet
#         # Assuming model_unet expects [batch_size, channels, height, width]
#         predicted_22nd_frame = predicted_22nd_frame.permute(0, 1,2, 3).to(device)

#         # Predict the semantic mask of the predicted 22nd frame
#         mask_output = modelunet2(predicted_22nd_frame)
#         predicted_mask = torch.argmax(mask_output, dim=1).squeeze(0)

#         # Store the predicted mask
#         predicted_masks_simvp[i] = predicted_mask

# # Verify the shape of the predicted masks tensor
# print(predicted_masks_simvp_hidden.shape)  # Should be [1000, 160, 240]

