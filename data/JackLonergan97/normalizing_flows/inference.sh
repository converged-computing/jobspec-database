#!/bin/bash
#PBS -N inference
#PBS -l nodes=1:ppn=1
#PBS -j oe
#PBS -o inference.log
#PBS -V
#PBS -t 1-100

# Change directory to the location from which this job was submitted
cd $PBS_O_WORKDIR
# Disable core-dumps (not useful unless you know what you're doing with them)
ulimit -c 0
export GFORTRAN_ERROR_DUMPCORE=NO
# Ensure there are no CPU time limits imposed.
ulimit -t unlimited
# Tell OpenMP to use all available CPUs on this node.
export OMP_NUM_THREADS=1
# Run Galacticus
source /home/jlonergan/data1/jlonergan/conda3/etc/profile.d/conda.sh
conda activate tensorflow
#python example_summary_statistic_distribution.py CDM_res8_no_subsample $PBS_ARRAYID > dg_message 2>&1
python subhalos_inference.py CDM_res8_no_subsample $PBS_ARRAYID > error_message 2>&1
