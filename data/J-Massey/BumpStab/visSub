#!/bin/bash
  
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --partition=amd
#SBATCH --time=03:50:00
#SBATCH --output=paraview.out
#SBATCH --error=paraview.out

module purge
module load conda
source activate an
module load paraview/5.11

mpiexec -n $SLURM_NPROCS pvserver --connect-id=11111 --displays=0
