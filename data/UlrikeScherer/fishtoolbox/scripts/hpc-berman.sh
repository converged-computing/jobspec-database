#!/bin/bash -l

#SBATCH --job-name=umap_clustering_berman
#SBATCH --output=res_%j.txt     # output file
#SBATCH --error=res_%j.err      # error file
#SBATCH --ntasks=1
#SBATCH --time=0-34:00 
#SBATCH --mem-per-cpu=30000      # memory in MB per cpu allocated
#SBATCH --partition=ex_scioi_gpu # partition to submit to
#SBATCH --gres=gpu:1

module load nvidia/cuda/10.0    # load required modules (depends upon your code which modules are needed)
module load comp/gcc/7.2.0

#source ./venv/bin/activate      # activate your python environment
#source ~/.condarc
source ~/miniconda3/etc/profile.d/conda.sh

echo "deactivating base env"
conda deactivate

echo "adding language encodings"
export LANG=UTF-8
export LC_ALL=en_US.UTF-8

echo "activating conda environment"
conda activate toolbox

echo "JOB START"

python hpc_berman.py

echo "JOB DONE"

echo "deactivating environments"
conda deactivate 

