#!/bin/bash
#PBS -l select=1:ncpus=40:ngpus=8
#PBS -l walltime=12:00:00
#PBS -q dgx
#PBS -N tf-resnet50
#PBS -j oe
### Specify project code
#PBS -P 99999999

# Tensorflow (Resnet 50) Imagenet
# Docker image to use
image="nvcr.io/nvidia/tensorflow:latest"
# 50 layer CNN
layers=50
# batch size
batch=256
#model
model=resnet
#dataset
dataset=imagenet

# Change to directory where job was submitted
if [ x"$PBS_O_WORKDIR" != x ] ; then
 cd "$PBS_O_WORKDIR" || exit $?
fi

# Location of dataset in Lustre
data="/scratch/projects/41000001/datasets/$dataset/$model"

stdout="$PWD/stdout.$PBS_JOBID"
stderr="$PWD/stderr.$PBS_JOBID"
nscc-docker run $image << EOF
cd /workspace/nvidia-examples/cnn
mpiexec --allow-run-as-root -np 8  python ${model}.py \
 --layers $layers \
 -b $batch \
 --data_dir $data > $stdout 2> $stderr
EOF
