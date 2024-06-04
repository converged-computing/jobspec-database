#!/bin/bash
#SBATCH --nodes=!NODES
#SBATCH --time=!WALL
#SBATCH --ntasks-per-node=!TASKS
#SBATCH --output=/scratch/r/rbond/msyriac/mpi_output_%j.txt
#SBATCH --cpus-per-task=!THREADS


cd $SLURM_SUBMIT_DIR

export DISABLE_MPI=false

module load NiaEnv/2022a
module load autotools
module load gcc/11.3.0 
module load openblas
module load gsl
module load openmpi
module load fftw
module load python

# module load NiaEnv/2019b
# module load intel/2019u4 
# export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=disable
# module load intelmpi/2019u4
# module load python/3.8.5
# module load autotools
# module load cfitsio
# module load gsl

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun !CMD 

