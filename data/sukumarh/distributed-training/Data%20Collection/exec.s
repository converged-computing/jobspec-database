#!/bin/bash
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=102400
#SBATCH --job-name=training_logger
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

module load python/intel/3.8.6 cuda/10.2.89 nccl/cuda10.2/2.7.8  

source ~/dl/bin/activate

python distributed_trainer.py -c "29,30,31,32,33,34" -w 16 -d data/ -s training_data/
