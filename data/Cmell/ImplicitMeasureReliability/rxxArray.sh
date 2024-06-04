#!/bin/bash

#SBATCH --partition shas
#SBATCH --job-name Run_3-MPI
#SBATCH --time 24:00:00
#SBATCH --nodes 10
#SBATCH --ntasks-per-node 24
#SBATCH --qos normal
#SBATCH --output mpiRun_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chme2908@colorado.edu

module purge
module load R/3.3.0
module load openmpi/1.10.2

DATE=$(date +%Y-%m-%d_%H.%M.%S)
let ncores=SLURM_NTASKS
let taskId=SLURM_ARRAY_TASK_ID
echo "Number of cores:" $ncores
ESTFL="Results/est_group${taskId}.RData"
mpiexec -np $ncores Rscript rxx031mpi.R --n.iter=10000 --dateStr=$DATE --guideFl=FullGuide.RData --estFlNm=$ESTFL --group=$taskId
