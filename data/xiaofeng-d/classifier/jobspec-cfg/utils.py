# utils.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

class MRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image and label, this will need to be modified to handle your data format
        image = np.load(self.image_paths[index])
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
def load_data(batch_size, data_paths):

    data_dir = data_paths["root"]

    # Load the saved tensors
    train_data_tensor = torch.load(os.path.join(data_dir, 'train_data_tensor.pt'))
    train_labels_tensor = torch.load(os.path.join(data_dir, 'train_labels_tensor.pt'))
    val_data_tensor = torch.load(os.path.join(data_dir, 'val_data_tensor.pt'))
    val_labels_tensor = torch.load(os.path.join(data_dir, 'val_labels_tensor.pt'))

    # Create TensorDataset objects
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader