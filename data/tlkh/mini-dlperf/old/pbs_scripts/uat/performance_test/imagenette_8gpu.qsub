#!/bin/bash
#PBS -l select=1:ncpus=40:ngpus=8
#PBS -l pmem=500GB
#PBS -N imagenette_8gpu
#PBS -j oe
#PBS -o imagenette_8gpu.log
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/tensorflow:20.02-tf2-py3

cd "$PBS_O_WORKDIR" || exit $?

nvidia-smi

export HOME=/home/users/uat

cd /home/users/uat/scratch
rm -rf imagenette_8gpu
mkdir imagenette_8gpu
pwd
ls
cd imagenette_8gpu
git clone --depth 1 https://github.com/tlkh/mini-dlperf
cd mini-dlperf/tensorflow
bash setup.sh

/usr/local/mpi/bin/mpirun --allow-run-as-root -np 8 \
  -bind-to none -map-by slot \
  -x LD_LIBRARY_PATH -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python3 cnn_train_hvd.py \
    --xla --amp --batch_size 512 --lr 0.8 \
    --dataset imagenette/320px --data_dir=/raid/tensorflow_datasets \
    --epochs 90 --img_aug --ctl

nvidia-smi
