#!/bin/bash -l

#SBATCH -J bench

#SBATCH -A 2017-12-20

#SBATCH --time=01:00:58
#SBATCH --time-min=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks=32

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=avmo@kth.se
#SBATCH -e SLURM.bench.%J.stderr
#SBATCH -o SLURM.bench.%J.stdout

N0=512
N1=512
N2=128

source /etc/profile
module load gcc/7.2.0
module swap PrgEnv-cray PrgEnv-intel
module swap intel intel/18.0.0.128
module add cdt/17.10 # add cdt module

export FLUID_PROC_MESH='2x32'

aprun -n $(nproc) \
       test_bench.out --N0=$N0 --N1=$N1 --N2=$N2
