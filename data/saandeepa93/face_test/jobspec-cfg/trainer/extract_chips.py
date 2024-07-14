import sys
sys.path.append('.')

import torch 
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# DISTRIBUTED COMPUTING
import webdataset as wds
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # loader
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group

from imports import * 
from utils import * 
from configs import get_cfg_defaults
from dataset import BGCDataset
from models import FaceFeatureModel
from supCon import SupConLoss




def ddp_setup(args):
  if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
  args.distributed = args.world_size > 1
  ngpus_per_node = torch.cuda.device_count()
  if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
      ic("DISTRIBTUED")
      args.rank = args.local_rank
      args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
      ic("SLURM")
      args.rank = int(os.environ['SLURM_PROCID'])
      args.gpu = int(os.environ['SLURM_LOCALID'])
    init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    if args.rank!=0:
      def print_pass(*args,  **kwargs):
          pass
      builtins.print = print_pass

    torch.cuda.set_device(args.gpu)
    torch.distributed.barrier()

def prepare_loader(cfg):
  train_dataset = BGCDataset(cfg, "train")
  val_dataset = BGCDataset(cfg, "val")
  
  if cfg.TRAINING.DISTRIBUTED:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, \
      num_workers=cfg.DATASET.NUM_WORKERS, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, \
      num_workers=cfg.DATASET.NUM_WORKERS, pin_memory=True, sampler=val_sampler)
  else:
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.DATASET.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.DATASET.NUM_WORKERS)

  return train_loader, val_loader


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert tensors to lists
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_images(process_data, mode):
  from torchvision.transforms import ToPILImage
  imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu().clamp(-1.,1.)) for vid in range(process_data.shape[1])  ]
  # imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu()) for vid in range(process_data.shape[1])  ]
  for id, im in enumerate(imgs):
    im.save(f"./data/vid_loader/{mode}_{id}_1.jpg")

if __name__ == "__main__":
  seed_everything(42)

  torch.autograd.set_detect_anomaly(True)
  torch.cuda.empty_cache()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/", f"{args.config}.yaml")

  # writer = SummaryWriter(f'./runs/{args.config}')

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()

  # DDP TRAINING
  if cfg.TRAINING.DISTRIBUTED:
    ddp_setup(args)

  # LOADER
  train_loader, val_loader = prepare_loader(cfg)
  
  output_shard_pattern = "./data/face_chips/close_range/shard-%06d.tar"
  sink =  wds.ShardWriter(output_shard_pattern, maxcount=10, maxsize=10e9, verbose=0)
  import io
  for b, (x_g, x_p, meta) in enumerate(tqdm(val_loader)):
    buffer_g = io.BytesIO()
    torch.save(x_g.contiguous().cpu().detach(), buffer_g)  # Move tensor to CPU before serialization
    image_bytes_g = buffer_g.getvalue()

    buffer_p = io.BytesIO()
    torch.save(x_p.contiguous().cpu().detach(), buffer_p)  # Move tensor to CPU before serialization
    image_bytes_p = buffer_p.getvalue()


    meta['probe_frames'] = [int(k.item()) for k in meta['probe_frames']]
    json_str = json.dumps(meta, cls=TensorEncoder)
    json_bytes = json_str.encode("utf-8")

    sample = {  
      "__key__": f"sample{b:06d}",
      "image_g.pt": image_bytes_g,
      "image_p.pt": image_bytes_p,
      "json": json_bytes
    }

    sink.write(sample)


  sink.close()
  with open('./data/face_chips/completed_tars_close_range_val.pkl', 'wb') as file:
    # Use pickle.dump() to serialize and save the list to the file
    pickle.dump(train_loader.dataset.completed, file)
  
  print("done!!")



