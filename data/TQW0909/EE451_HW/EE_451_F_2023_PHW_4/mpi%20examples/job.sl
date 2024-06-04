#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=00:05:00
#SBATCH --output=mpijob.out
#SBATCH --constraint="xeon-4116"
#SBATCH --error=mpijob.err

module purge
module load gcc/8.3.0
module load openmpi/4.0.2
module load pmix/3.1.3

export UCX_TLS=sm,tcp,self

ulimit -s unlimited

srun --mpi=pmix_v2 -n $SLURM_NTASKS ./go

