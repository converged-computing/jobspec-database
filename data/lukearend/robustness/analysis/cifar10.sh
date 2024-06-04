#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --job-name=cifar10
#SBATCH --output=out/cifar10-0000%a.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

SCALE_FACTOR=('0.25' '0.5' '1' '2' '4' '0.25' '0.5' '1' '2' '4')
BATCH_NORM_FLAG=('' '' '' '' '' '--disable_batch_norm' '--disable_batch_norm' '--disable_batch_norm' '--disable_batch_norm' '--disable_batch_norm')

module load openmind/singularity/older_versions/2.4

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/analysis/activations.py \
--model_dir=/om/user/larend/models/robust/cifar10/0000${SLURM_ARRAY_TASK_ID} \
${BATCH_NORM_FLAG[$SLURM_ARRAY_TASK_ID]} \
--scale_factor=${SCALE_FACTOR[$SLURM_ARRAY_TASK_ID]} \
--dataset=cifar10 \
--pickle_dir=/om/user/larend/robustpickles/cifar10/0000${SLURM_ARRAY_TASK_ID} \
--host_filesystem=/om

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/analysis/redundancy.py \
--pickle_dir=/om/user/larend/robustpickles/cifar10/0000${SLURM_ARRAY_TASK_ID}

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/analysis/robustness.py \
--model_dir=/om/user/larend/models/robust/cifar10/0000${SLURM_ARRAY_TASK_ID} \
${BATCH_NORM_FLAG[$SLURM_ARRAY_TASK_ID]} \
--scale_factor=${SCALE_FACTOR[$SLURM_ARRAY_TASK_ID]} \
--dataset=cifar10 \
--pickle_dir=/om/user/larend/robustpickles/cifar10/0000${SLURM_ARRAY_TASK_ID} \
--host_filesystem=/om
