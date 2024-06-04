#!/bin/bash
#
#! -cwd
#! -j y
#! -S /bin/bash

# Name of the job
#SBATCH --job-name=mergeSi2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH --partition=med2                 # Use the high partition
#SBATCH -t 0-00:30                      # Runtime in D-HH:MM format
#SBATCH -o 'outputs/cSiaSiMD-%j.output' #File to which STDOUT will be written
#SBATCH --mail-user="dgunruh@ucdavis.edu"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=33-64

# run one thread for each one the user asks the queue for
# hostname is just for debugging
# hostname
export OMP_NUM_THREADS=1
# export t=$SLURM_ARRAY_TASK_ID
export j=$SLURM_JOB_ID
export t=$SLURM_ARRAY_TASK_ID
module load openmpi

# assign the random seed and the output files for the lammps scripts
s=$j   # 124248+$j+$t
temp=aSi${t}
log=aSi/${temp}
addendum=55
beginning=inputs/neb_input_${temp}.out
ending=inputs/neb_end_${temp}.out
idfile=inputs/neb_atoms_${temp}.out
dump=neb_calc_result_${temp}_${s}_${t}
pNum=32

srun ../lammps_3Mar2020/build/lmp_mpi -var i $beginning -var f $ending -var pNum $pNum -var idfile $idfile -var dumpfile $dump -log logfiles/${log}/log.lammps -screen screenfiles/screen${temp} -partition ${pNum}x2 -in nebGAP.in
