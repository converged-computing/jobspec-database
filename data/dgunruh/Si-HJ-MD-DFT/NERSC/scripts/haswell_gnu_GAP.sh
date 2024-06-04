#!/bin/bash
# Name of the job
#SBATCH --job-name=aSiGAP

# Submission details
#SBATCH --qos=regular
#SBATCH --constraint=haswell
# SBATCH --time=10
#SBATCH --nodes=4
#SBATCH --tasks-per-node=32 # fastest is 2 cpus per task (haswell has 2 cpus per core). I would do more but then we run into memory issues
#SBATCH --cpus-per-task=2
#SBATCH --mem=118G   # max memory haswell is 118G, knl is 87G
#SBATCH -t 1-12:00                      # Runtime in D-HH:MM format
#SBATCH --array=0-9

# Outputs
#SBATCH -o 'outputs/cSiaSiMD-%j.output' #File to which STDOUT will be written
#SBATCH --mail-user="dgunruh@ucdavis.edu"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# allocate the openMP threads
# export OMP_PROC_BIND=true
# export OMP_PLACES=threads
export OMP_NUM_THREADS=1

j=$SLURM_JOB_ID
t=$SLURM_ARRAY_TASK_ID

pe=${PE_ENV,,}
if [ $pe != "gnu" ]; then
	module swap PrgEnv-$pe PrgEnv-gnu
fi

# module swap PrgEnv-intel PrgEnv-gnu
# module load lammps
# module load python3/3.6.0
# module load openmpi

# assign the random seed and the output files for the lammps scripts
s=$((j+100*t))
dumpA=aSi-GAP-$j-$t.xyz
dumpsnapA=aSiBox-GAP-$j-$t.xyz

srun --cpu_bind=cores /global/common/software/m3634/lammps_3Mar2020/gnubuild_haswell/lmp_gnu_haswell -var s $s -var d $dumpA -var ds $dumpsnapA -in createAmorphousSi.in
