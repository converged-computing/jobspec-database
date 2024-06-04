#!/bin/bash
#SBATCH --job-name=kpc-dask-2node
#SBATCH --partition=regular2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=3
#SBATCH --time=06:00:00


module purge
module load  gnu8/8.3.0  openmpi3/3.1.4 


source /home/znazari/.bashrc

conda activate Zainab-env #you have to change this with your environment

cd $SLURM_SUBMIT_DIR

mpirun -n 3 python  main_dask_algorithm.py > time.txt

