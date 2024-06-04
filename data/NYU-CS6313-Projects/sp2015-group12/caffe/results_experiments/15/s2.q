#!/bin/bash
#PBS -V
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=20:00:00
#PBS -l mem=2GB
#PBS -N rfnbus
#PBS -M rfn216@nyu.edu
 
module purge
module load h5py/intel/2.3.1
module load caffe/intel/20150115
RUNDIR=$SCRATCH/test/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR
#DATADIR=$SCRATCH/my_project/data
cd $RUNDIR

python /home/rfn216/BusRio/split_by_bus_line_date2.py > out.log 2>&1

exit
