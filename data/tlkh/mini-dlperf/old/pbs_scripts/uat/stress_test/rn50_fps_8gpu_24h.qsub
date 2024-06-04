#!/bin/bash
#PBS -l select=1:ncpus=40:ngpus=8
#PBS -l pmem=500GB
#PBS -N rn50_fps_8gpu
#PBS -j oe
#PBS -o rn50_fps_8gpu.log
#PBS -v CONTAINER_IMAGE=nvcr.io/nvidia/tensorflow:20.02-tf2-py3

cd "$PBS_O_WORKDIR" || exit $?

nvidia-smi

export HOME=/home/users/uat

cd /home/users/uat/scratch
rm -rf rn50_fps_8gpu
mkdir rn50_fps_8gpu
pwd
ls
cd rn50_fps_8gpu
git clone --depth 1 https://github.com/tlkh/mini-dlperf
cd mini-dlperf/tensorflow
bash setup.sh

# 1000 steps -> 440s
# 24h -> 196 epochs

/usr/local/mpi/bin/mpirun --allow-run-as-root -np 8 \
  -bind-to none -map-by slot \
  -x LD_LIBRARY_PATH -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python3 cnn_train_hvd.py \
    --xla --amp --batch_size 512 \
    --dataset imagenette/160px --data_dir=/raid/tensorflow_datasets \
    --epochs 196 --ctl --steps 1000 --no_val --verbose 2

nvidia-smi
