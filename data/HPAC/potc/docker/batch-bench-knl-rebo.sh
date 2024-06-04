#!/usr/local_rwth/bin/zsh
#SBATCH --mem-per-cpu=1024M
#SBATCH --job-name=knl_potc_bench
#SBATCH --output=results/knl/log-%J.txt
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --partition=c16k

set -e
set -u
set -x

module load gcc/6
module switch intel intel/18.0
module switch openmpi intelmpi

base=$(mktemp -d -p $TEMP)
echo $base
mkdir $base/lammps
time cp -rp $PWD/nodocker/lammps/src $base/lammps
mkdir $base/lammps/lib
time cp -rp $PWD/nodocker/lammps/lib/kokkos $base/lammps/lib
mkdir $PWD/results/knl-rebo/$SLURM_JOB_ID
lammps=$base/lammps potc=$PWD/.. tmp=$PWD/results/knl-rebo/$SLURM_JOB_ID ./benchmark.sh test-intel-rebo-regular.sh
rm -r $base
