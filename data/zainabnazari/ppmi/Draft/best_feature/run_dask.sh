#!/bin/bash
#SBATCH --job-name=kpc-dask-2node
#SBATCH --partition=regular2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=10000
#SBATCH --time=04:00:00


module purge
module load gnu11 openmpi3  


source /home/znazari/.bashrc

conda activate Zainab-env #you have to change this with your environment

cd $SLURM_SUBMIT_DIR

mpirun -n 6 python main_dask_algorithm.py 

