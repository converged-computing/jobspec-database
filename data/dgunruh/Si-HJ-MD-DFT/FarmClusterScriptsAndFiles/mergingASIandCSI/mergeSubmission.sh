#!/bin/bash
#
#! -cwd
#! -j y
#! -S /bin/bash

# Name of the job
#SBATCH --job-name=cSiaSiGAP
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH -c 2
#SBATCH --mem=60G
#SBATCH --partition=high                 # Use the high partition
#SBATCH -t 2-00:00                      # Runtime in D-HH:MM format
#SBATCH -o 'outputs/cSiaSiMD-%j.output' #File to which STDOUT will be written
#SBATCH --mail-user="dgunruh@ucdavis.edu"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# run one thread for each one the user asks the queue for
# hostname is just for debugging
# hostname
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export t=$SLURM_ARRAY_TASK_ID
export j=$SLURM_JOB_ID

module load openmpi

# assign the random seed and the output files for the lammps scripts
s=$j   # 124248+$j+$t
dumpA=aSi-GAP-$j.xyz
dumpsnapA=aSiBox-GAP-$j.xyz

srun ../src/lammps-stable_3Mar2020/build/lmp_mpi -var s $s -var d $dumpA -var ds $dumpsnapA -in mergeAmorphousCrystallineGAP.in
