#!/bin/sh
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --partition=amd
#SBATCH --time=06:50:00
#SBATCH --output=pview.out
#SBATCH --error=pview.out

module purge
module load paraview/5.10.1 python/3.11

# mpiexec pvserver --connect-id=11111 --egl-device-index=1 : pvserver --egl-device-index=1
mpiexec -n $SLURM_NPROCS pvserver --connect-id=11111 --displays=0
