#!/bin/bash
# Example slurm running script.
#SBATCH -A GR79-29
#SBATCH -p gpu
#SBATCH	--cpus-per-task=1
#SBATCH	--ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=[1-200]
#SBATCH --output=slurm%A_%a.out
#SBATCH	--time=2-00:00:00
#SBATCH --mail-user=mkmkl93@gmail.com
#SBATCH	--mail-type=ALL
#SBATCH	--mem=90000
#SBATCH --nodelist=rysy-n6

module load gpu/cuda/10.2 common/compilers/gcc/8.3.1
python3 ~/nasze-ca/src/prot-gen.py ${SLURM_ARRAY_TASK_ID}
