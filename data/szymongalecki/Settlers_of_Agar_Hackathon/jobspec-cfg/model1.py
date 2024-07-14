import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score
import torchvision
from torch.utils.data import random_split
import torchvision
import numpy as np


def eval(y_true, y_pred):
    score = accuracy_score(y_true, y_pred) * recall_score(y_true, y_pred)
    return score


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, threshold=True):
        self.data_dir = data_dir
        self.transform = transform
        self.threshold = threshold
        self.image_paths = [
            os.path.join(data_dir, filename)
            for filename in os.listdir(data_dir)
            if filename.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.data_dir, f"{image_name}.json")

        image = Image.open(image_path).convert("RGB")

        with open(label_path, "r") as f:
            label_data = json.load(f)

        colonies_number = label_data["colonies_number"]

        if self.threshold:
            colonies_number = min(colonies_number, 1)

        if self.transform:
            image = self.transform(image)

        return image, colonies_number


data_dir = "train_data_resized"

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
    ]
)

dataset = CustomDataset(data_dir, transform=transform, threshold=True)

dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

num_epochs = 100

learning_rate = 0.001

model = torchvision.models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    c = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        labels = labels.unsqueeze(1)
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        threshold = 0.5
        predicted = (outputs >= threshold).int()

        train_correct += (predicted.flatten() == labels.flatten()).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    official_score = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        labels = labels.unsqueeze(1)
        loss = criterion(outputs.float(), labels.float())

        val_loss += loss.item() * images.size(0)

        threshold = 0.5
        predicted = (outputs >= threshold).int()
        val_correct += (predicted.flatten() == labels.flatten()).sum().item()
        val_total += labels.size(0)
        official_score += eval(labels.flatten().cpu(), predicted.flatten().cpu())

    avg_val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total
    avg_official_score = official_score / len(val_loader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Official Score: {avg_official_score}"
    )
    if avg_official_score > best_val:
        best_val = avg_official_score
        torch.save(model.state_dict(), "agar3000.pth")
