#! /bin/bash

#SBATCH -t 5
#SBATCH -p q_student_gpu
#SBATCH --gpus=2
# #SBATCH --gres=gpu:tesla:1
# #SBATCH --gres=gpu:ampere:1

. vars_in.sh

BINARY="../src/juliaset_gpu"

# change these bsizes according to your best values
BSIZE_X=32
BSIZE_Y=1

for SIZE in "${SIZE_LIST[@]}"
do 
  $BINARY -r $SIZE $SIZE -b $BSIZE_X $BSIZE_Y -n $NREP -o "task4_gpu_${SIZE}.csv"
done

