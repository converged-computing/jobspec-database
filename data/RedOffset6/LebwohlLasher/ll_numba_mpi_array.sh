#!/bin/bash

#SBATCH --account=PHYS030544
#SBATCH --job-name=ll_mpi_numba_array
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=0:30:00
#SBATCH --array=1-10

echo 'running mpi test'

module load languages/miniconda

echo "Before activation: $(which python)"

#/mnt/storage/software/languages/miniconda/miniconda3.11.3-CompressAI-Vision0.1.3/condabin/conda activate desktop_clone_env
#conda init bash
#possibly I need to run "conda init bash" every time i open a new terminal
source activate desktop_clone_env

echo "After activation: $(which python)"

#module load languages/intel/2020-u4

size=$((SLURM_ARRAY_TASK_ID*75))

srun --mpi=pmi2 python LebwohlLasher_mpi_numba.py 50 $size 0.5 0

