#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --job-name=iVAE
#SBATCH --workdir=/proj/barimpac/beyond-NonSENS/iVAE/
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 1-0  # running time
#SBATCH --mem=10G

module purge
module load Python/3.6.6-foss-2018b
module load cuDNN/7.5.0.56-CUDA-10.0.130

source /wrk/users/barimpac/research/bin/activate

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

# Executable here
#python main.py --config discrete-full-ivae.yaml --n-sims 1 --m 0.5 --ckpt_folder='run/checkpoints/'
#jupyter lab --no-browser --port=8889
#srun python3 main.py --config binary-10-lbfgs.yaml --n-sims 1 --m 1.0 --s 3 --ckpt_folder='run/checkpoints/'
srun python3 main.py --config binary-5-identity-adam-Copy9.yaml --n-sims 3 --m 2.0 --s 3 --ckpt_folder='run/checkpoints/'
