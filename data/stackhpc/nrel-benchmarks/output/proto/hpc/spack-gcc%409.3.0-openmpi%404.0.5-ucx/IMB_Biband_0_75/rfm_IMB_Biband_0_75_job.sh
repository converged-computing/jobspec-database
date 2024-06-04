#!/bin/bash
#SBATCH --job-name="rfm_IMB_Biband_0_75_job"
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=12
#SBATCH --output=rfm_IMB_Biband_0_75_job.out
#SBATCH --error=rfm_IMB_Biband_0_75_job.err
#SBATCH --time=0:10:0
#SBATCH --exclusive
#SBATCH --partition=hpc
export SLURM_MPI_TYPE=pmix_v3
spack load intel-mpi-benchmarks
srun IMB-MPI1 biband -npmin 1
