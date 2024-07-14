import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from SentencesDataset import SentencesDataset
from CBOW import CBOW
from Train import train_multi_gpu

def collate_func(batch):
  contexts = []
  targets = []
  for i in range(0, len(batch)):
    contexts += batch[i][0]
    targets += batch[i][1]
  return contexts, targets

def ddp_setup(rank: int, world_size: int):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355' # Check on clusterDEI

  # initialize the process group
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

def cleanup():
  dist.destroy_process_group()


def main(rank: int,
         world_size: int,
         dataset_path='dataset_manipulation/it_polished.txt',
         context_size=2,
         embedding_dim=300,
         epochs=10,
         batch_size=32,
         save_every=5):

  ddp_setup(rank, world_size)

  # Creating the dataset
  print('Creating the dataset...')
  dataset = SentencesDataset(dataset_path, context_size)
  dataset.print_info()

  # Creating the dataloader
  print('Creating the dataloader...')
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_func,
    pin_memory=True,
    sampler=DistributedSampler(dataset)
  )

  # Creating the model
  print('Creating the model...')
  model = CBOW(len(dataset.word_to_idx), embedding_dim, context_size)

  # Training the model
  print('Training the model...')
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = torch.nn.NLLLoss()
  for epoch in range(epochs):
    train_multi_gpu(
      rank,
      model, dataloader, optimizer, criterion, save_every, epoch, dataset.idx_to_word,
    )


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Distributed Data Parallel Training")
  parser.add_argument('-d', '--dataset_path', type=str, default='dataset_manipulation/it_polished.txt')
  parser.add_argument('-c', '--context_size', type=int, default=2)
  parser.add_argument('-e', '--embedding_dim', type=int, default=300)
  parser.add_argument('-ep', '--epochs', type=int, default=10)
  parser.add_argument('-b', '--batch_size', type=int, default=32)
  parser.add_argument('-s', '--save_every', type=int, default=5)

  main_args = (
    parser.parse_args().dataset_path,
    parser.parse_args().context_size,
    parser.parse_args().embedding_dim,
    parser.parse_args().epochs,
    parser.parse_args().batch_size,
    parser.parse_args().save_every,
  )

  print(main_args)

  world_size = torch.cuda.device_count()
  print('Number of GPUs: {}'.format(world_size))
  mp.spawn(main,
           args=(world_size, *main_args),
           nprocs=world_size)
