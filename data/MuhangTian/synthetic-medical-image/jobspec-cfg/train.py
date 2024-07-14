import argparse
import time

import yaml

import wandb
from model.ddpm import GaussianDiffusion, Trainer, Unet


def parse_arguments():
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("--project_name", type=str, default="public-image-data")
    prs.add_argument("--entity", type=str, default="muhang-tian")
    prs.add_argument("--dim", type=int, default=64)
    prs.add_argument("--image_size", type=int, default=128)
    prs.add_argument("--timesteps", type=int, default=1000)
    prs.add_argument("--sampling_timesteps", type=int, default=1000)
    prs.add_argument("--objective", type=str, default="pred_noise", choices=['pred_noise', 'pred_x0', 'pred_v'])
    prs.add_argument("--wandb", action='store_true')
    prs.add_argument("--beta_schedule", type=str, choices=["linear", "cosine", "sigmoid"], default="sigmoid")
    prs.add_argument("--channels", type=int, default=3)
    prs.add_argument("--resnet_block_groups", type=int, default=8)
    prs.add_argument("--learned_sinusoidal_dim", type=int, default=16)
    prs.add_argument("--attn_dim_head", type=int, default=32)
    prs.add_argument("--attn_heads", type=int, default=4)
    prs.add_argument("--load_path", type=str, required=True)
    prs.add_argument("--batch_size", type=int, default=32)
    prs.add_argument("--lr", type=float, default=8e-5)
    prs.add_argument("--num_steps", type=int, default=700000)
    prs.add_argument("--gradient_accumulate_every", type=int, default=2)
    prs.add_argument("--ema_decay", type=float, default=0.995)
    prs.add_argument("--save_and_sample_every", type=int, default=1000)
    prs.add_argument("--num_fid_samples", type=int, default=5000)
    prs.add_argument("--sweep_id", type=str, default=None)
    args = prs.parse_args()
    return args

def train(args, run_id: str=None, sweep: bool=True):
    if sweep:
        run = wandb.init()
        args = wandb.config
        sub_dir = "sweep"
        run_id = run.id
        wandb.log({"run_id": run_id})
        print(f"run_id: {run_id}")
    else:
        sub_dir = "runs"
    
    model = Unet(
        dim = args.dim,
        channels = args.channels,
        resnet_block_groups = args.resnet_block_groups,
        learned_sinusoidal_dim = args.learned_sinusoidal_dim,
        attn_dim_head = args.attn_dim_head,
        attn_heads = args.attn_heads,
        dim_mults = (1, 2, 4, 8),
        full_attn = (False, False, False, True),
        flash_attn = True,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.image_size,
        timesteps = args.timesteps,           # number of steps
        sampling_timesteps = args.sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = args.objective,
        beta_schedule = args.beta_schedule,
        auto_normalize = True,
    )

    trainer = Trainer(
        diffusion,
        args.load_path,
        train_batch_size = args.batch_size,
        train_lr = args.lr,
        train_num_steps = args.num_steps,
        gradient_accumulate_every = args.gradient_accumulate_every,
        ema_decay = args.ema_decay,
        save_and_sample_every = args.save_and_sample_every,
        results_folder = f"./results/{sub_dir}/{run_id}",
        num_fid_samples=args.num_fid_samples,
        wandb = wandb,
        amp = True,        
        calculate_fid = True,
    )

    trainer.train()
    
if __name__ == "__main__":
    args = parse_arguments()
    
    if args.sweep_id is not None:
        wandb.agent(
            args.sweep_id, 
            function=lambda: train(args=None, sweep=True), 
            entity=args.entity, 
            project=args.project_name
        )
    else:
        run_id = str(int(time.time()))
        if args.wandb:
            wandb.init(
                project=args.project_name, 
                entity=args.entity, 
                name=run_id, 
                config=args
            )
        else:
            wandb = None
        train(args, sweep=False, run_id=run_id)
        