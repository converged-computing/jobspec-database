#!/bin/bash
#SBATCH --output=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-%j.out
#SBATCH --error=/home/mila/m/marawan.gamal/projects/tensor-net/outputs/slurm-error-%j.out
#SBATCH --mem=10G                                         # Ask for 10 GB of RAM
#SBATCH --time=8:00:00                                    # The job will run for 8 hours
#SBATCH -x cn-g[005-012,017-026]

# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/mila/m/marawan.gamal/scratch/.venv/rosa/bin/activate

# 3. Copy your dataset on the compute node
#cp -r /home/mila/m/marawan.gamal/.cache/huggingface $SLURM_TMPDIR/huggingface

#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=wic train.epochs=10 train.batch_size=32 fnmodel.name=none train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=qnli train.epochs=10  train.batch_size=16 fnmodel.name=ia3 fnmodel.params.rank=2 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=qnli train.epochs=10  train.batch_size=16 fnmodel.name=ia3 fnmodel.params.rank=2 train.lr=2e-4
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=qnli train.epochs=10  train.batch_size=16 fnmodel.name=ia3 fnmodel.params.rank=2 train.lr=2e-3

# NVIDIA A100-SXM4-40GB
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 train.lr=2e-5
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 fnmodel.name=rosa +task=cola fnmodel.params.rank=4 train.epochs=4 train.lr=2e-3
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 fnmodel.name=lora +task=cola fnmodel.params.rank=4 train.epochs=4 train.lr=2e-4

#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 +task=cola train.epochs=5 train.lr=2e-5 train.optimizer.name=sgd
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 fnmodel.name=rosa fnmodel.params.factorize_mode=bottom +task=cola fnmodel.params.rank=2 train.epochs=5 train.lr=2e-3 fnmodel.params.bias_requires_grad=False train.optimizer.name=sgd fnmodel.factorize_freq=1
#python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface +profile=marawan seed=42 fnmodel.name=lora +task=cola fnmodel.params.rank=2 train.epochs=5 train.lr=2e-3 fnmodel.params.bias_requires_grad=False train.optimizer.name=sgd


python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface seed=42 +profile=marawan +task=cola train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 fnmodel.params.factorize_method=random_proj train.lr=2e-3
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface seed=42 +profile=marawan +task=cola train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 fnmodel.params.factorize_method=random_proj train.lr=2e-4
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface seed=42 +profile=marawan +task=cola train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 fnmodel.params.factorize_method=random_proj train.lr=2e-5
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface seed=42 +profile=marawan +task=cola train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 fnmodel.params.factorize_method=random_proj_orthogonal train.lr=2e-3
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface seed=42 +profile=marawan +task=cola train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 fnmodel.params.factorize_method=random_proj_orthogonal train.lr=2e-4
python train_mlm.py dataset.cache=$SLURM_TMPDIR/huggingface seed=42 +profile=marawan +task=cola train.batch_size=16 fnmodel.name=rosa fnmodel.params.rank=2 fnmodel.params.factorize_method=random_proj_orthogonal train.lr=2e-5


# Wandb sweep commands
# wandb sweep --project rosa-mlm-sweep sweep_mlm.yaml
# wandb agent tensor-lab/rosa-mlm-sweep/6isjw8pm --count 5
