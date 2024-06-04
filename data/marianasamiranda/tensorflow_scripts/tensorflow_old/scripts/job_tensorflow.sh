#!/bin/bash

#SBATCH -J tf-tenosorflow      # Job name
#SBATCH -o tf-tenosorflow.o%j  # Name of stdout output file
#SBATCH -e tf-tenosorflow.e%j  # Name of stderr error file
#SBATCH -p rtx                 # Queue (partition) name
#SBATCH -N 1                   # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                   # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 03:00:00            # Run time (hh:mm:ss)

WORKSPACE=$(dirname $(dirname $(realpath $0)))
DATA_DIR="/scratch1/08486/mmiranda/mysharedirectory"

#echo "cp ${DATA_DIR}/objects/44g.zip /dev/shm"
#cp "${DATA_DIR}/objects/44g.zip" /dev/shm

#echo "cp ${DATA_DIR}/objects/7g.zip /dev/shm"
#cp "${DATA_DIR}/objects/7g.zip" /dev/shm

#echo "cp ${DATA_DIR}/objects/6g.zip /dev/shm"
#cp "${DATA_DIR}/objects/6g.zip" /dev/shm

#echo "cp ${DATA_DIR}/objects/3g /dev/shm"
#cp "${DATA_DIR}/objects/3g" /dev/shm

#echo "module load cuda/10.1 cudnn/7.6.5 nccl/2.5.6"
#module load cuda/10.1
#module load cudnn/7.6.5
#module load nccl/2.5.6

#echo "module load remora"
#module load remora

cd "${WORKSPACE}/scripts"

EPOCHS=1
BATCH_SIZE=256
DATE="$(date +%Y_%m_%d-%H_%M)"
TARGET_DIR="/tmp"

# 100g

DATASET="/home/gsd/tensorflow/100g_tfrecords"

for i in {1..1}; do
#  RUN_DIR="${TARGET_DIR}/lenet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}"  $RUN_DIR

#  RUN_DIR="${TARGET_DIR}/alexnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
#
  RUN_DIR="${TARGET_DIR}/resnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  #remora ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 1 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  sleep 10
  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
done

# 200g

DATASET="${DATA_DIR}/imagenet_processed/200g_2048_tfrecords"

#for i in {1..1}; do
#  RUN_DIR="${TARGET_DIR}/lenet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
# 
#  RUN_DIR="${TARGET_DIR}/alexnet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
# 
#  RUN_DIR="${TARGET_DIR}/resnet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
#done
