#!/bin/bash
#PBS -j oe
#PBS -N life_hpc_n8
###PBS -l select=1:ncpus=1:ngpus=1:mem=5gb
#PBS -l select=8:ncpus=1:ngpus=1:mem=5gb
##PBS -q debug
#PBS -q parallel
#PBS -A CON13_INFN          
module load pgi openmpi/1.6.5--pgi--14.1

#ulimit -m
cd /eurora/home/userexternal/ralfieri/life/openacc/game-of-life

cat $PBS_NODEFILE

mpicc -O3 life_hpc.c -acc -DCOMP -DMPI -ta=nvidia -Minfo=accel -lpgacc -o life_hpc_acc_mpi

for N in $(echo 8 4 2 1)
do
for DIM in $(echo 17000 8000 4000 2000 1000 )
do 
for COMP in $(echo 1000 100 10 0)
do
CMD="mpirun -np $N ./life_hpc_acc_mpi -r $DIM -c $DIM -n $COMP -s 10 -d 0"
echo "# $CMD"
eval $CMD
done
done
done
