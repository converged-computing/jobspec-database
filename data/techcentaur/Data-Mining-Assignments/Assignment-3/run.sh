#!/bin/sh
### Set the job name (for your reference)
#PBS -N Run_1
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=4:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=00:10:00

#PBS -l software=python
# After job starts, must goto working directory.
# $PBS_O_WORKDIR is the directory from where the job is fired.
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

module load compiler/python/3.6.0/ucs4/gnu/447
module load pythonpackages/3.6.0/numpy/1.16.1/gnu
module load pythonpackages/3.6.0/networkx/2.1/gnu
module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
module load lib/blas/openblas/0.2.19/gnu
module load lib/blas/3.6.0/gnu
module load lib/blas/netlib/3.7.0/gnu
module load lib/blas/openblas/0.2.19/gnu
module load lib/openblas/0.2.20/gnu
module load lib/opencv/3.2.0/gnu

#job
python main.py > output.txt
