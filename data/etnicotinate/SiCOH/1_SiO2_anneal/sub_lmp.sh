#!/bin/bash
#PBS -N lmp_test
#PBS -l select=4:ncpus=8:mem=8G:mpiprocs=2:ompthreads=4
###PBS -l select=8:ncpus=128:mem=440G:mpiprocs=128:ompthreads=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o pbs.log
# Users are advised to always specify the Project ID accessible to users while submitting a job.
#PBS -P personal-e0945881
#PBS -q normal

cd $PBS_O_WORKDIR;
echo "nodefile: $PBS_NODEFILE"

module load lammps/23Jun2022_update1-b1
# openmpi/4.1.2-hpe
module load openmpi

input_file=new.lmp
# mpirun -n 8 lmp -in ${input_file} 1>  ${input_file%.lmp}.log 2> error.log
mpirun --hostfile $PBS_NODEFILE lmp -in ${input_file} 1>  ${input_file%.lmp}.log 2> error.log

