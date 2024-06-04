#!/bin/bash -l

#PBS -t 1-{0}
#PBS -l walltime={1}
#PBS -l nodes={2}:ppn={3}
#PBS -l mem={4}
#PBS -j oe
#PBS -N {5}
#PBS -o log/{5}

# Execute the line matching the array index from file {6}:
# 
cmd=`head -${{PBS_ARRAYID}} ~/infoh413/{6} | tail -1`

# Execute the command extracted from the file:
cd ~/infoh413

module load python/3.4.0a1
module load openmpi/1.6.5/gcc/4.8.2
module load R/3.0.2
module unload gcc/4.6.1/gcc/4.4.7 || true
module unload gdb/7.7/gcc/4.4.7 || true

make nocolor=true

eval $cmd
