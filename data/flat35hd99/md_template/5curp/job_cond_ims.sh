#!/bin/sh
#PBS -l select=1:ncpus=40:mpiprocs=8:ompthreads=5:jobtype=small
#PBS -l walltime=167:00:00
prefix=$TEMPLATE_PROJECT_PREFIX

if [ "$PBS_O_WORKDIR" ]; then
  cd ${PBS_O_WORKDIR}
fi

module purge
source $prefix/util/load_curp_ims

set -xeu

output=$prefix/5curp/output/$group_type/${run}_${smp}
# faster directory
tmp=/ramd/users/$USER/$PBS_JOBID

cd $tmp

echo "job start time" `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

cp $output/flux.nc .
cp $0 $output/

echo "copy end time " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

mpirun -np 8 \
  curp cal-tc \
    --frame-range 1 50000 1 --average-shift 2 \
    --dt 0.001  \
    -a acf.nc \
    -o $output/conductivity.dat \
    flux.nc > $output/conductivity.log

echo "cond end time " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

cp acf.nc $output/

echo "copy end time " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log
