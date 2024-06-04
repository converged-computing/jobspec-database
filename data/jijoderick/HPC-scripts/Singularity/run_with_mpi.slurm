#!/bin/bash
#SBATCH --job-name=JDAHFpEF
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=24:00:00
#SBATCG --mail-type=ALL
#SBATCH --mail-user=jijoderick.abraham@uq.net.au
module load mpi/openmpi-x86_64
mpirun singularity exec /home/s4657117/HCM-project/SingularitY/hcm-project.img python Python_filename.py
