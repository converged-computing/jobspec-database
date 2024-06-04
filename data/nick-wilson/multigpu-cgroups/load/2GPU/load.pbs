#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=1:00:00
#PBS -q fj5
#PBS -P 41000001
#PBS -N cgroup_test
#PBS -j oe
cd "$PBS_O_WORKDIR" || exit $?
image="nvcr.io/nvidia/mxnet:18.02-py2"
cd "$PBS_O_WORKDIR" || exit $?
nvidia-docker-run $image < stdin-1 > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID
