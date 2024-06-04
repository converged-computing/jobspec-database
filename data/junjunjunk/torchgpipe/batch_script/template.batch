#!/bin/sh
#PBS -q l-small
#PBS -l select=2:mpiprocs=1:ompthreads=1
#PBS -W group_list=xx00
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR

GROUP=$(id -ng)
MYDIR=/lustre/${GROUP}/${USER}

source ${MYDIR}/.bashrc
conda activate torchgpipe

module purge
cd ./torchgpipe/benchmarks/resnet101-speed
module load cuda10/10.0.130

python main.py pipeline-2 -b 256 -c 4  > ./logs/pipeline2_b256_c4.log 2>&1

