#!/bin/bash
#SBATCH --job-name parallel-DDP-2GPUs 
#SBATCH -D . 
#SBATCH --output %j.out
#SBATCH --error %j.err
#SBATCH --ntasks-per-node=1    
#SBATCH -n 1                
#SBATCH -c 80                  
#SBATCH --gres='gpu:2'         
#SBATCH --time 01:00:00

module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML arrow/3.0.0 torch/1.9.0a0 text-mining/2.1.0

python -m torch.distributed.launch --nproc_per_node=2 main_ddp.py --json_file './config.json'
 
