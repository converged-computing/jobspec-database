#!/bin/bash -l
#SBATCH --job-name="no_spillover"
#SBATCH --time=16:00:00
#SBATCH --nodes=25
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

#SBATCH -e eta_error_file.e
#SBATCH -o eta_output_file.o

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for i in $(seq 0 5 50)
do
   srun python3 ./eta_mpi.py $i
done
