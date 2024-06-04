#!/bin/zsh
#SBATCH -p cca
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=8
#SBATCH --constraint='rome,ib'
#SBATCH --reservation='rocky8'
#SBATCH --job-name='pysrtune'
#SBATCH --time=7-00:00:00

module purge && module load modules/2.0-20220630 cuda/11.4.4 cudnn/8.2.4.15-11.4 gcc/10.3.0 python/3.10.4 ffmpeg/4.4.1-nix imagemagick git/2.35.1 texlive singularity zsh/5.8 disBatch/beta

for i in {1..30}; do
    srun --exclusive -N 1 --ntasks 1 /bin/zsh -c 'source ~/.zshrc && cd /mnt/home/mcranmer/pysr_wandb/ && wandb agent mcranmer/pysr_wandb/51ga3lt4' &
done

wait