#!/bin/bash
#SBATCH -J plasma-python           # Job name
#SBATCH -o plasma.o%j       # Name of stdout output file
#SBATCH -e plasma.e%j       # Name of stderr error file
#SBATCH -p gpu        # Queue (partition) name
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH -n 20               # Total # of mpi tasks (should be 1 for serial)
#SBATCH --mail-user=michoski@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A Magnetic-Confinement       # Allocation name (req'd if you have more than 1)

module load gcc/4.9.3
module load python3/3.5.2
module load cuda/8.0
module load cudnn/5.1
module load tensorflow-gpu/1.0.0
module load mvapich2
module load git

#remove checkpoints for a benchmark run
#rm /scratch/gpfs/$USER/model_checkpoints/*
#rm /scratch/gpfs/$USER/results/*
#rm /scratch/gpfs/$USER/csv_logs/*
#rm /scratch/gpfs/$USER/Graph/*
#rm /scratch/gpfs/$USER/normalization/*

#export OMPI_MCA_btl="tcp,self,sm"
ibrun python3 mpi_learn.py