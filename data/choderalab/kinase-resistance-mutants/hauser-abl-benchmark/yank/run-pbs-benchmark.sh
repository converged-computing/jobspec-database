#!/bin/bash
#    Begin PBS directives
#PBS -A chm126
#PBS -N resistance
#PBS -j oe
#PBS -l walltime=0:20:00,nodes=1
#PBS -l gres=atlas1%atlas2
#PBS -l feature=gpudefault
#    End PBS directives and begin shell commands
# MANUAL STEP: Change the path to the output of your `pwd`
# MANUAL STEP: Change the path to the output of your `pwd`
export HOME=/lustre/atlas/scratch/jchodera1/chm126
export MINICONDA="$HOME/miniconda3"
export PATH="$MINICONDA/bin:$PATH"
export LD_LIBRARY_PATH=$MINICONDA/lib:$LD_LIBRARY_PATH
cd /lustre/atlas/scratch/jchodera1/chm126/kinase-resistance-mutants/hauser-abl-benchmark/yank/
module load cudatoolkit
export OPENMM_CUDA_COMPILER=`which nvcc`
# Set up mpi environment
module remove PrgEnv-pgi
module add PrgEnv-gnu
module add cray-mpich
# Run only one job to set things up
export OE_LICENSE="/lustre/atlas/scratch/jchodera1/chm126/.openeye/oe_license.txt"
#aprun -n $PBS_NUM_NODES yank script --yaml=allmuts-sams.yaml --jobid="ALPS_APP_PE" --njobs="PBS_NUM_NODES"
#aprun -n $PBS_NUM_NODES yank script --yaml=allmuts-sams.yaml
aprun -n $PBS_NUM_NODES python benchmark.py
