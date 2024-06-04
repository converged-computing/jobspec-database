#!/bin/bash
#
#! -cwd
#! -j y
#! -S /bin/bash

# Name of the job
#SBATCH --job-name=mergeSi2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
# SBATCH -c 1
#SBATCH --mem=250G
#SBATCH --partition=med2                 # Use the high partition
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
displacement=2.97
change=3.14
dumpA=bulk_heatstrip-$displacement-$change.xyz
dumpsnapA=box_heatstrip-$displacement-$change.xyz

srun ../lammps_3Mar2020/build/lmp_mpi -var s $s -var disp $displacement -var cng $change -var d $dumpA -var ds $dumpsnapA -in heatStrip.in
