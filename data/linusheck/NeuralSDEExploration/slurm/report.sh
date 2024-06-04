#!/bin/bash

#SBATCH --job-name=analyze
#SBATCH --account=tipes
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --output=/home/linushe/outputs/%x.%A_%4a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linus.heck@rwth-aachen.de

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Some initial setup
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge
module load julia
module add texlive

/home/linushe/neuralsdeexploration/slurm/latestfile.sh | xargs julia --project=/home/linushe/neuralsdeexploration /home/linushe/neuralsdeexploration/scripts/generate_report.jl

