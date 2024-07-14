import sys
sys.path.append('.')
import io
import braceexpand
import tarfile

import torch 
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from collections import defaultdict

# DISTRIBUTED COMPUTING
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # loader
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group
from torchvision.transforms import ToPILImage
import torch.distributed as dist

import webdataset as wds
from transforms import GroupNormalize, ToTorchFormatTensor, Stack
from einops import rearrange

from imports import * 
from utils import * 
from configs import get_cfg_defaults
from dataset2 import BGCDataset
from models import FaceFeatureModel
from supCon import SupConLoss




def ddp_setup(args):
  if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
  args.distributed = args.world_size > 1
  ngpus_per_node = torch.cuda.device_count()
  if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
      args.rank = args.local_rank
      args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
      args.rank = int(os.environ['SLURM_PROCID'])
      args.gpu = int(os.environ['SLURM_LOCALID'])
    init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    # if args.rank!=0:
    #   def print_pass(*args,  **kwargs):
    #       pass
    #   builtins.print = print_pass

    torch.cuda.set_device(args.gpu)
    
def custom_collate_fn(batch):
  x_g = batch[0]
  x_p = batch[1]
  print(x_p.size(), x_g.size())
  return torch.utils.data.dataloader.default_collate(batch)
  batched_json = {
        "gallery": [],
        "probe": [],
        "probe_frames": []
  }
  json_data = [item[2] for item in batch]
  for item in json_data:
    batched_json["gallery"].extend(item["gallery"])
    batched_json["probe"].append(item["probe"])
    batched_json["probe_frames"].extend(item["probe_frames"])
  
  x1_batch = torch.stack([x1 for x1, x2, _ in batch])
  x2_batch = torch.stack([x2 for x1, x2, _ in batch])
  return x1_batch, x2_batch, batched_json 


def preprocess(batch):

  scale = lambda x: (x - x.min()) / (x.max() - x.min())

   # JSON PREPROCESSING
  json_bytes = batch['json']
  json_dict = json.loads(json_bytes.decode('utf-8'))

  # GALLERY PREPROCESSING
  data1 = batch['image_g.pt']
  buffer1 = io.BytesIO(data1)
  tensor1 = torch.load(buffer1)
  x_g = tensor1[:, :, 1]

  # PROBE PREPROCESSING
  data2 = batch['image_p.pt']
  buffer2 = io.BytesIO(data2)
  tensor2 = torch.load(buffer2)
  x_p = tensor2[:, :, 1]
  
 
  all_subjects = [k.split('/')[1] for k in json_dict['probe']]
  all_fields = [k.split('/')[2] for k in json_dict['probe']]

  batch['image_g.pt'] = x_g
  batch['image_p.pt'] = x_p
  batch['json'] = all_fields
  # batch['json'] = torch.tensor(labels)
  return batch

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src

def prepare_loader(cfg, args):

  train_url = "./data/face_chips/shard-{000000..000010}.tar"
  val_url = "./data/face_chips/shard-{000011..000014}.tar"

  # train_url = "./data/face_chips/shard-{000000..001037}.tar"
  # val_url = "./data/face_chips/val/shard-{000000..000200}.tar"

  if cfg.TRAINING.DISTRIBUTED:
    train_urls = list(braceexpand.braceexpand(train_url))
    val_urls = list(braceexpand.braceexpand(val_url))

    train_ds_size = len(train_urls) * 10
    val_ds_size = len(val_urls) * 10

      # .map(preprocess)\
    train_dataset = wds.WebDataset(train_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess)\
      .to_tuple("image_g.pt", "image_p.pt", "json")
      
      # .map(preprocess)\
    val_dataset = wds.WebDataset(val_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess)\
      .to_tuple("image_g.pt", "image_p.pt", "json")
  else:
    train_dataset = wds.WebDataset(train_url)
    val_dataset = wds.WebDataset(val_url)

    train_dataset = train_dataset\
            .map(preprocess)\
            .to_tuple("image_g.pt", "image_p.pt", "json")\
            .shuffle(5000)\
            .batched(cfg.TRAINING.BATCH_SIZE, partial=False)
    val_dataset = val_dataset\
              .map(preprocess)\
              .to_tuple("image_g.pt", "image_p.pt", "json")\
              .shuffle(0)\
              .batched(cfg.TRAINING.BATCH_SIZE, partial=False) 


  if cfg.TRAINING.DISTRIBUTED:
    world_size =  dist.get_world_size()
    
    train_n_batches =  max(1, train_ds_size // (cfg.TRAINING.BATCH_SIZE * world_size))
    train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0)
    train_loader = train_loader.unbatched().shuffle(1000).batched(cfg.TRAINING.BATCH_SIZE).with_epoch(train_n_batches)

    val_n_batches =  max(1, val_ds_size // (cfg.TRAINING.BATCH_SIZE * world_size))
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)
    val_loader = val_loader.unbatched().shuffle(0).batched(cfg.TRAINING.BATCH_SIZE).with_epoch(val_n_batches)
    # train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0)
    # val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)

  else:
    train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)

  return train_loader, val_loader


def save_images(process_data, mode):
  imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu().clamp(-1.,1.)) for vid in range(process_data.shape[1])  ]
  # imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu()) for vid in range(process_data.shape[1])  ]
  for id, im in enumerate(imgs):
    im.save(f"./data/vid_loader/{mode}_{id}_1.jpg")


def filter_tar_files(directory, target_distance):
    selected_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".tar"):
            tar_path = os.path.join(directory, filename)
            # distance = extract_metadata(tar_path)
            # if distance == target_distance:
            #     selected_files.append(tar_path)
    return selected_files




def extract_distance(tar_path):
  openface_root = "/shares/rra_sarkar-2135-1003-00/faces/openface_feats_extractor/data/openface/train/probe"
  distances = []
  openface_dict = []
  tf = tarfile.open(tar_path)
  for member in tf.getmembers():
    if "json" in member.name:
      f = tf.extractfile(member)
      if f is not None:
          metadata_content = f.read()
          metadata = json.loads(metadata_content)
          distance = metadata['probe'][0].split("/")[2]
          if distance in ['close_range', 'closerange']:
            distance = 'close_range'
          openface_csv = os.path.join(openface_root, metadata['probe'][0])
          if os.path.isdir(openface_csv):
            openface_dict.append(distance)
          distances.append(distance)
  
  return distances, openface_dict

if __name__ == "__main__":
  seed_everything(42)

  torch.autograd.set_detect_anomaly(True)
  torch.cuda.empty_cache()
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/", f"{args.config}.yaml")


  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()

  # # DDP TRAINING
  # if cfg.TRAINING.DISTRIBUTED:
  #   ddp_setup(args)
  #   ic("RANK: ", args.rank, args.gpu)

  # LOADER
  # train_loader, val_loader = prepare_loader(cfg, args)

  tar_dir = "./data/face_chips"
  tar_list = [os.path.join(tar_dir, f"shard-{str(k).zfill(6)}.tar") for k in range(2000)]
  # ic(tar_list)

  distance_counts_per_tar = {}
  openface_counts_per_tar = {}
  for tar_path in tar_list:
    distances, openface_distances = extract_distance(tar_path)

    distance_counts = {distance: distances.count(distance) for distance in set(distances)}
    distance_counts_per_tar[tar_path] = distance_counts
  
    openface_counts = {distance: openface_distances.count(distance) for distance in set(openface_distances)}
    openface_counts_per_tar[tar_path] = openface_counts
  
  sum_dict = {}
  # Iterate through each record in the nested dictionary
  for key, inner_dict in distance_counts_per_tar.items():
      # Iterate through each item in the inner dictionary
      for inner_key, value in inner_dict.items():
          # If the inner key is already in sum_dict, add the value to its current sum
          if inner_key in sum_dict:
              sum_dict[inner_key] += value
          # Otherwise, initialize the sum for this inner key
          else:
              sum_dict[inner_key] = value

  ic(sum_dict)

  sum_dict2 = {}
  # Iterate through each record in the nested dictionary
  for key, inner_dict in openface_counts_per_tar.items():
      # Iterate through each item in the inner dictionary
      for inner_key, value in inner_dict.items():
          # If the inner key is already in sum_dict, add the value to its current sum
          if inner_key in sum_dict2:
              sum_dict2[inner_key] += value
          # Otherwise, initialize the sum for this inner key
          else:
              sum_dict2[inner_key] = value

  ic(sum_dict2)
  e()
  
    
  # Initialize a dictionary to keep track of the total counts for each distance
  total_distance_counts = defaultdict(int)

  # List to keep track of selected tar files
  selected_tar_files = []

  # Sort tar files by the smallest maximum count of any distance to prioritize balance
  sorted_tar_files = sorted(distance_counts_per_tar.items(), key=lambda item: min(item[1].values()))

  for tar_path, distance_counts in sorted_tar_files:
    # Check if adding this tar file improves balance
    improved_balance = True
    for distance, count in distance_counts.items():
      if total_distance_counts[distance] + count > (min(total_distance_counts.values()) + count):
        improved_balance = False
        break
      
      if improved_balance:
        # Update total counts and add tar file to the selection
        for distance, count in distance_counts.items():
          total_distance_counts[distance] += count
        selected_tar_files.append(tar_path)
  ic(selected_tar_files)