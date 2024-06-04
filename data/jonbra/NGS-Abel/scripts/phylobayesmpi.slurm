#!/bin/bash
#SBATCH --job-name=pbmpiORG1
#SBATCH --account=nn9404k
#SBATCH --output=slurm-%j.base
#SBATCH --time=600:00:00
#SBATCH --ntasks=61
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=long
#SBATCH --mail-type=all

STR="$(ls *.phy -x1)"
module load phylobayesmpi
mpirun -n 61 pb_mpi -d "$STR" -cat -gtr -dc "$STR".pb
