import sys
sys.path.append('.')
import os 
import io 
import json 

import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

from decord import VideoReader, cpu
import webdataset as wds

from icecream import ic 
from sys import exit as e 
from transforms import * 
import pickle
from tqdm import tqdm





class DataAugmentationForVideoMAE(object):
  def __init__(self):
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupScale((224, 224))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        # ADD ANY AUG BEFORE THIS LINE
        ToTorchFormatTensor(div=True),
    ])

  def __call__(self, images):
    process_data = self.transform(images)
    return process_data

  def __repr__(self):
    repr = "(DataAugmentationForVideoMAE,\n"
    repr += "  transform = %s,\n" % str(self.transform)
    repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    repr += ")"
    return repr




def preprocess(batch):
  transform = DataAugmentationForVideoMAE()

  data = batch['image_g.pt']
  buffer = io.BytesIO(data)
  tensor = torch.load(buffer).squeeze()
  x_g = tensor[:, :100]


  data = batch['image_p.pt']
  buffer = io.BytesIO(data)
  tensor = torch.load(buffer).squeeze()
  x_p = tensor[:, :]

  # meta_g = batch['label_g.cls'].decode("utf-8")
  # meta_p = batch['label_p.cls'].decode("utf-8")
  json_bytes = batch['json']
  json_dict = json.loads(json_bytes.decode('utf-8'))

  batch['image_g.pt'] = x_g
  batch['image_p.pt'] = x_p
  batch['json'] = json_dict
  return batch




if __name__ == "__main__":
  
  url = "./data/face_chips/shard-000000.tar"
  
  dataset = wds.WebDataset(url)
  dataset = dataset.shuffle(5)\
            .map(preprocess)\
            .to_tuple("image_g.pt", "image_p.pt", "json")
  
  
  dataloader = DataLoader(dataset, num_workers=0, batch_size=1)

  for x_g, x_p, meta in dataloader:
    print(meta)
    print(x_g.size())
    print(x_p.size())

    # imgs = [ ToPILImage()(x_g[0][:, vid, :, :].cpu().clamp(0,0.996)) for vid in range(x_g.shape[2])  ]
    # for id, im in enumerate(imgs):
    #   im.save(f"./data/vid_loader/gallery_{id}.jpg")
    # imgs = [ ToPILImage()(x_p[0][:, vid, :, :].cpu().clamp(0,0.996)) for vid in range(x_p.shape[2])  ]
    # for id, im in enumerate(imgs):
    #   im.save(f"./data/vid_loader/probe_{id}.jpg")
    e()