#!/bin/bash
#PBS -V
#-o /dev/null
#-e /dev/null
#PBS -l pmem=124mb,walltime=23:00:00

pwd
python run_queue.python $PBS_JOBNAME

