#!/bin/bash
# Name of the job
#SBATCH --job-name=cSiaSiMD

# Submission details
#SBATCH --qos=regular
#SBATCH --constraint=haswell
# SBATCH --time=10
#SBATCH --nodes=4
#SBATCH --tasks-per-node=32 # fastest is 2 cpus per task (haswell has 2 cpus per core)
#SBATCH --cpus-per-task=2
#SBATCH --mem=118G   # max memory haswell is 118G, knl is 87G
#SBATCH -t 1-12:00                      # Runtime in D-HH:MM format
#SBATCH --array=0-14

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
if [ $pe != "intel" ]; then
	module swap PrgEnv-$pe PrgEnv-intel
fi
# module swap PrgEnv-gnu PrgEnv-intel
# module load lammps
# module load python3/3.6.0
# module load openmpi

# assign the random seed and the output files for the lammps scripts
s=$((j + 100*t))
dumpA=aSi-GAP-$j-$t.xyz
dumpsnapA=aSiBox-GAP-$j-$t.xyz

srun --cpu_bind=cores /global/common/software/m3634/lammps_3Mar2020/intelbuild/lmp_intel -var s $s -var d $dumpA -var ds $dumpsnapA -in createAmorphousSiIntel.in
