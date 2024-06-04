#!/usr/bin/env bash

#PBS -N MG3_KERNEL 
#PBS -A NTDD0004
#PBS -l select=1:ncpus=36:mpiprocs=36:mem=300GB
#PBS -l walltime=00:59:00
#PBS -q casper 
#PBS -j oe
#PBS -k eod

ulimit -s unlimited

# unload any modules currently loaded
module purge

# load modules
module load ncarenv/1.3
module load gnu/9.1.0
module load openmpi/4.1.0
module load ncarcompilers/0.5.0

export PATH=/glade/work/sunjian/llvm/12.0.1/bin:$PATH
export LD_LIBRARY_PATH=/glade/work/sunjian/llvm/12.0.1/lib:$LD_LIBRARY_PATH

cd $PBS_O_WORKDIR
let pcol=384   # pcols value for input data
let ntask=36   # ntasks value for input data

# nvidia-cuda-mps-control -d && echo "MPS control daemon started"

# add a loop to compile the code with different number of mpi ranks
for n in 1 
do
    # add a loop to compile the code with different dfact
    for i in 1  # 1 2 4 8 16 32 64 128 256 512
    do
        # compile the code
        make clean
        make ntasks=$ntask pcols=$pcol dfact=$i
    
        # run the code
        mpirun -n $n ./kernel.exe >& casper_llvm_mpiranks${n}_pcols${pcol}_dfact${i}_log
    
        # clean the files
        make clean
    
    done # loop for i

done # loop for n
