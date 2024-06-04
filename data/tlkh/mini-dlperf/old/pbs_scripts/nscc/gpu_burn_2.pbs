#!/bin/sh

#PBS -l select=1:ncpus=10:ngpus=2

#PBS -l walltime=2:00:00

#PBS -q dgx

#PBS -N GPU_BURN_2

#PBS -j oe

#PBS -o gpu_burn_2.log

cd "$PBS_O_WORKDIR" || exit $?

image="nvcr.io/nvidia/tensorflow:20.02-tf2-py3"

nscc-docker run $image << EOF

nvidia-smi
export HOME=/home/users/sutd/1002653 
export TFDS_DIR=/scratch/users/sutd/1002653/tensorflow_datasets
cd /scratch/users/sutd/1002653/mini-dlperf && \
git pull && \
cd tensorflow && \
ls && \
bash setup.sh && \
echo $TFDS_DIR && \
python3 gpu_burn.py --num_gpus 2 --stats && \
python3 gpu_burn.py --num_gpus 2

EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

