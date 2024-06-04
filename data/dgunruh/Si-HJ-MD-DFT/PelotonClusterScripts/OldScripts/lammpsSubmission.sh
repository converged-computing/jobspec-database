#!/bin/bash -l

# Name of the job
#SBATCH --job-name=cSiaSiMD
# SBATCH --nodes=2
# SBATCH --ntasks=2
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --partition=med2
#SBATCH --time=24:00:00
#SBATCH --output='outputs/cSiaSiMD-%j.output'
#SBATCH --mail-user="dgunruh@ucdavis.edu"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# run one thread for each one the user asks the queue for
# hostname is just for debugging
# hostname
# export OMP_NUM_THREADS=$SLURM_NTASKS
export t=$SLURM_ARRAY_TASK_ID
module load lammps

# the main job executable to run: note the use of srun before it
# srun lmp_serial -in cSiaSi_workingVersion.in
# mpirun lmp_mpi -in cSiaSi_workingVersion.in

# assign the random seed and the output files for the lammps scripts
s=21248+100*$t
dumpA=aSi-$t.xyz
dumpsnapA=aSiBox-$t.xyz
dumpI=cSiaSiInterface-$t.xyz
dumpsnapI=cSiaSiInterfaceSnapshot-$t.xyz

# sed -rie 's/(rand_seed equal)\s\w+/\1 $s/gi' createAmorphousSi.in
# sed -rie 's/(rand_seed equal)\s\w+/\1 $s/gi' mergeAmorphousCrystalline.in
mpirun lmp_mpi -var s $s -var d $dumpA -var ds $dumpsnapA -in createAmorphousSi.in
mpirun lmp_mpi -var s $s -var d $dumpI -var dA $dumpA -var ds $dumpsnapI -in mergeAmorphousCrystalline.in
