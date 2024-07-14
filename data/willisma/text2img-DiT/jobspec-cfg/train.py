# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from einops import rearrange
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from functools import partial

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from transformers import DistilBertModel
from coco import CocoDataset, collate_fn
from wandb_utils import initialize_wandb, log_loss_dict, log_images


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

@torch.no_grad()
def text_encoding(caption, encoder, end_token, dropout=None):
    device = caption.device
    mask = torch.cumsum((caption == end_token), 1).to(device)
    mask[caption == end_token] = 0
    mask = (~mask.bool()).long()

    emb = encoder(caption, attention_mask=mask)['last_hidden_state']
    
    emb = rearrange(emb, 'b c h -> b (c h)')

    return emb

def update_model_state(model_state, state_dict_path, fine_tuning=False):

    pretrained_state = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
    ckpt_steps = 0

    if fine_tuning:
        opt_state = None
    else:
        opt_state = pretrained_state["opt"]
        pretrained_state = pretrained_state["model"]

        path = state_dict_path.split(os.sep)
        ckpt_steps = int(path[-1][0:-3])

    for name, param in pretrained_state.items():
        if name not in model_state:
                continue
        elif isinstance(param, torch.nn.parameter.Parameter):
            param = param.data

        if fine_tuning:
            '''
                Load all weights but AdaLN; 
                Freeze all weights but Bias and LayerNorm
            '''
            if "adaLN" not in name:
                model_state[name].copy_(param)
            if "bias" not in name or "adaLN" not in name:
                model_state[name].requires_grad = False
        else:
            model_state[name].copy_(param)
        
        # if fine_tuning == True and "adaLN_modulation" not in name:
        #     model_state[name].requires_grad = False
    
    return model_state, opt_state, ckpt_steps

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        wandb_configs = {
            "model": model_string_name,
            "epochs": args.epochs,
            "learning_rate": 1e-4,
            "batch_size": args.global_batch_size,
            "GPUs": dist.get_world_size(),
            "checkpoint_path": args.ckpt,
        }
        initialize_wandb(wandb_configs, exp_name=f"{model_string_name}-{experiment_index}-{args.cfg_scale}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=1000,
        emb_dropout_prob=0.1,
    )

    model_state = None
    opt_state = None

    # Load pretrained state
    ckpt_steps = 0 # steps loaded from checkpoint
    if args.ckpt is not None:
        assert os.path.isfile(args.ckpt), f'Could not find DiT checkpoint at {args.ckpt}'
        
        model_state, opt_state, ckpt_steps = update_model_state(model.state_dict(), args.ckpt, fine_tuning=args.fine_tuning)
        model.load_state_dict(model_state)

        print("Model Checkpoint loaded successfully")

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule, MSE loss
    test_diffusion = create_diffusion(str(250)) # for sampling
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    logger.info(f"DiT optimizer: learning rate {1e-4}")

    # if opt_state is not None:
    #     opt.load_state_dict(opt_state)
    #     print("Optimizer Checkpoint loaded successfully")
    
    # Setup data:
    assert os.path.isdir(args.data_path), f'Could not find COCO2017 at {args.data_path}'

    train_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    train_path = os.path.join(args.data_path, 'train2017')
    train_ann_path = os.path.join(args.data_path, 'annotations/captions_train2017.json')

    train_dataset = CocoDataset(train_path, train_ann_path, transform=train_transform)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        collate_fn=collate_fn,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            y = text_encoding(y, encoder, 102)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y) # class conditional
            # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0 and train_steps > 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

                if rank == 0:
                    log_loss_dict({"Average Loss": avg_loss, "Steps / Sec": steps_per_sec}, train_steps)
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                if rank == 0:
                    torch.cuda.empty_cache() # reduce memory fragmentation

                    logger.info(f"Start Sampling with {y.shape[0]} samples")

                    n = y.shape[0]
                    z = torch.randn(n, 4, latent_size, latent_size, device=device)

                    z = torch.cat([z, z], 0)
                    y_null = torch.zeros_like(y).to(device)
                    y = torch.cat([y, y_null], 0)

                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale) # --> param for penalize unconditional output

                    # Sample images:
                    samples = test_diffusion.p_sample_loop(
                        ema.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )

                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                    images = []

                    for img in samples:
                        img = vae.decode(img.unsqueeze(0) / 0.18215).sample.detach().cpu()
                        images.append(img.squeeze())
                    images = torch.stack(images)

                    log_images(images, args.model.replace("/", "-"), train_steps)

                    logger.info("Sampling Done.")
                
                dist.barrier()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{(train_steps + ckpt_steps):07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="t2i-results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--fine-tuning", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    args = parser.parse_args()
    main(args)
