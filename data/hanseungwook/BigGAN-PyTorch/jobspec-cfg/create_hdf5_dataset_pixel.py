""" Convert dataset to HDF5
    This script preprocesses a dataset and saves it (images and labels) to 
    an HDF5 file for improved I/O. """
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm, trange
import h5py as h5

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import utils

def prepare_parser():
  usage = 'Parser for ImageNet HDF5 scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--image_size', type=int, default=256,
    help='Image size')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--output_dir', type=str, default='./',
    help='Default location where hdf5 will be saved (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=256,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  return parser


def run(config):
  # Get dataset
  kwargs = {'num_workers': config['num_workers'], 'pin_memory': False, 'drop_last': True}
  batch_size = config['batch_size']
  device = config['device']
  filters = utils.create_filters(device)

  norm_mean = [0.5,0.5,0.5]
  norm_std = [0.5,0.5,0.5]

  # Create transforms
  train_transform = transforms.Compose([utils.CenterCropLongEdge(), 
                     transforms.Resize(config['image_size']),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=norm_mean, std=norm_std)])

  train_dataset = ImageFolder(root=config['data_root'], transform=train_transform)
  train_loader = DataLoader(train_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            **kwargs)


  with h5.File(config['output_dir'] + '/ILSVRC_D64.hdf5', 'w') as f:
      x_ds = f.create_dataset('imgs', shape=(len(train_dataset), 3, 64, 64), dtype=np.float32, fillvalue=0)
      y_ds = f.create_dataset('labels', shape=(len(train_dataset), ), dtype=np.int64)

      for i, (x, y) in enumerate(tqdm(train_loader)):
          x_ds[i*batch_size:(i+1)*batch_size] = x.numpy()
          y_ds[i*batch_size:(i+1)*batch_size] = y.numpy()


def main():
  # parse command line and run    
  parser = prepare_parser()
  config = vars(parser.parse_args())

  # Use GPU, if available
  config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print(config)
  run(config)

if __name__ == '__main__':    
  main()