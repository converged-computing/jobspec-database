#!/bin/sh -eu
#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM --mpi proc=1
#PJM -S
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=1:jobtype=gpu:ngpus=1
#PBS -l walltime=4:00:00

prefix=$TEMPLATE_PROJECT_PREFIX

# hostname of Flow Type 2 calculation nodes is cx{number}
if [[ "$(hostname)" =~ cx ]];then
    specify_job_id=$(printf %03d $PJM_BULKNUM)

    module purge
    source $prefix/util/load_amber_gpu
    PMEMD=pmemd.cuda

    work=/local/${PJM_JOBID}[${PJM_BULKNUM}]
else
    # specify_job_id is provide by job submitter

    module purge
    source $prefix/util/load_amber_on_ims
    PMEMD=pmemd.cuda

    work=/ramd/users/$USER/$PBS_JOBID/$RANDOM
    mkdir -p $work
fi

run=${specify_job_id:0:2}
smp=${specify_job_id:2:1}

output=$prefix/4nve/output/${run}_${smp}

mkdir -p $output

# Restart file
## if smp = 0, trajectory_num = 5000000
## if smp = 3, trajectory_num = 20000000
trajectory_num=$(((smp+1)*5000000))
restart_file=$prefix/3sampling/output/$run/sampling/md.rst_${trajectory_num}

echo "job start time" `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

${PMEMD} \
  -O \
  -i $prefix/4nve/input \
  -p $prefix/0structure/output/system.prmtop \
  -c $restart_file \
  -ref $prefix/0structure/output/system.crd \
  -r $output/md.rst \
  -o $output/md.out \
  -x $work/md.crd.nc \
  -v $work/md.vel.nc \
  -inf $output/md.info \
  -l $output/md.log

echo "md end time   " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

cp $work/md.crd.nc $output/
cp $work/md.vel.nc $output/

echo "job end time  " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log
