#!/bin/bash
#SBATCH --job-name="rfm_Sysinfo_job"
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=rfm_Sysinfo_job.out
#SBATCH --error=rfm_Sysinfo_job.err
#SBATCH --time=0:10:0
#SBATCH --partition=hpc
export SLURM_MPI_TYPE=pmix_v3
srun python sysinfo.py
echo Done
