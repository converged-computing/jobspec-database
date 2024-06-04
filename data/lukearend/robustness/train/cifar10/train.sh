#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --job-name=train
#SBATCH --output=out/%a.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

module load openmind/singularity/older_versions/2.4

singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
python /om/user/larend/robust/train/cifar10/train.py \
--model_index="${SLURM_ARRAY_TASK_ID}" --host_filesystem=/om
