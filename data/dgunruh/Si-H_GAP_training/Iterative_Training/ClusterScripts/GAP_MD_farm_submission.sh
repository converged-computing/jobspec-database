#!/bin/bash
#
#! -cwd
#! -j y
#! -S /bin/bash

# Name of the job
#SBATCH --job-name=GAP_MD
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=64
# SBATCH -c 1
#SBATCH --mem=250G
#SBATCH --partition=med2                 # Use the high partition
#SBATCH -t 2-00:00                      # Runtime in D-HH:MM format
#SBATCH --array=31-35

#SBATCH -o 'farm_outputs/cSiaSiMD-%j.output' #File to which STDOUT will be written
#SBATCH --mail-user="dgunruh@ucdavis.edu"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# run one thread for each one the user asks the queue for
# hostname is just for debugging
# hostname
export OMP_NUM_THREADS=1
t=$SLURM_ARRAY_TASK_ID
j=$SLURM_JOB_ID
operation=$1 # options are: optimize, lowAnneal, medAnneal, highAnneal
phase=$2 # options are: amorph, liquid, vacancy, divacancy, interstitial
folder=$3
module load openmpi

# assign the random seed and the output files for the lammps scripts
s=$((j+100*t))   # 124248+$j+$t
inputfile=inputs/${folder}/GAP_${operation}_${phase}_${t}.xyz
dump=dumpseriesoutputs/${folder}/GAP_${operation}_${phase}_dumpseries$t.xyz
dumpsnap=dumpoutputs/${folder}/GAP_${operation}_${phase}_dump$t.xyz

srun ../lammps_3Mar2020/build/lmp_mpi -var s $s -var i $inputfile -var d $dump -var ds $dumpsnap -in ${operation}.in
