import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# Load CSV metadata
metadata_path = '/home/data_shares/geocv/pano_metadata.csv'
df = pd.read_csv(metadata_path)
print(df.head())
df["File Path"] = df["Photo ID"].apply(lambda x: f"{x}.jpg")
# Add file path to DataFrame
print(df.head())
df["File Path"] = df["File Path"].apply(lambda x: os.path.join('/home/data_shares/geocv/panos/', x))
df = df[["File Path", "Climate Zone", "Country", "Geocell"]]
print(df.head())
# Split DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(train_df.head())
print(val_df.head())
# Create Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print(train_dataset)
print(val_dataset)
# # Load images

# def load_images(batch):
    
#     for file_path in batch["file_path"]:
#         img = Image.open(file_path)
#         # Preserve other information in the dataset
#         data = {key: batch[key] for key in batch.keys()}
#         data["image"] = img
#         images.append(data)
#     return images

# train_dataset = train_dataset.map(load_images, batched=True)
# val_dataset = val_dataset.map(load_images, batched=True)

print(train_dataset)
print(val_dataset)
# Save split datasets
split_datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

print(split_datasets)

split_datasets.save_to_disk('/home/data_shares/geocv/panos_dataset.hf')

print("Datasets split and saved successfully!")
