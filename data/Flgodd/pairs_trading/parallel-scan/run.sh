#!/bin/bash
#PBS -q ampereq
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS  ngpus=1
#PBS  output=para.out

module load intel-parallel-studio-xe/compilers/64
mpirun hostname


echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`
echo GPU number: $CUDA_VISIBLE_DEVICES


module use /software/x86/tools/nvidia/hpc_sdk/modulefiles
module load nvhpc/22.9

#! Run the executable
#./d2q9-bgk input_128x128.params obstacles_128x128.dat
#./d2q9-bgk input_256x256.params obstacles_256x256.dat
#./d2q9-bgk input_1024x1024.params obstacles_1024x1024.dat
nvprof  ./parallel_scan

