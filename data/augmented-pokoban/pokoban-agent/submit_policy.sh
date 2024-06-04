#!/bin/sh
# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N A3C_unsupervised
# –- specify queue --
#PBS -q hpc
# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=24:00:00
# –- number of processors/cores/nodes --
#PBS -l nodes=1:ppn=20
# –- user email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#PBS -M arydbirk@gmail.com
# –- mail notification –-
#PBS -m abe
# -- run in the current working (submission) directory --
if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi
# here follow the commands you want to execute

module load python3/3.5.1
source /appl/tensorflow/1.3cpu-python3.5/bin/activate

/appl/glibc/2.17/lib/ld-linux-x86-64.so.2  --library-path /appl/glibc/2.17/lib/:/appl/gcc/4.8.5/lib64/:/usr/lib64/atlas:/lib64:/usr/lib64:$LD_LIBRARY_PATH $(which python) main_policy.py

