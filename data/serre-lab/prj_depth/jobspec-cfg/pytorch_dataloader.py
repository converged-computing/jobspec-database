import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle

class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_detail_path):
        self.image_dir = image_dir
        
        # Load depth details
        with open(depth_detail_path, 'rb') as f:
            depth_details = pickle.load(f)
            
        self.depth_diff_arr = depth_details['depth_diff_arr']
        self.lower_depth_range = depth_details['lower_depth_range']
        self.upper_depth_range = depth_details['upper_depth_range']
        
        # Filter image indices based on depth_diff range
        self.valid_indices = [i for i, diff in enumerate(self.depth_diff_arr) 
                              if self.lower_depth_range[0] <= diff <= self.lower_depth_range[1] or 
                                 self.upper_depth_range[0] <= diff <= self.upper_depth_range[1]]
        
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        img_path = f"{self.image_dir}/{img_idx}.png"
        img = Image.open(img_path)
        img = self.transform(img)
        
        depth_diff = self.depth_diff_arr[img_idx]
        if self.lower_depth_range[0] <= depth_diff <= self.lower_depth_range[1]:
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)
        
        return img, label

# Create the dataset and dataloader
dataset = DepthDataset("/cifs/data/tserre_lrs/projects/prj_depth/dataset/blender_images_all/",
                       "/cifs/data/tserre_lrs/projects/prj_depth/dataset/blender_images_all/depth_details.pkl")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_idx, (images, labels) in enumerate(dataloader):
    if batch_idx == 0:  # Only for the first batch
        max_values = images.max()  # Max values for each image in the batch
        min_values = images.min()  # Min values for each image in the batch
        
        print("Shape of images in the first batch : ", images.shape)
        print("Shape of labels in the first batch : ", labels.shape)
        print(f"Max values for images in the first batch: {max_values}")
        print(f"Min values for images in the first batch: {min_values}")
        print(f"Labels for images in the first batch: {labels}")
        break