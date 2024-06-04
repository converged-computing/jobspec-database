#!/bin/bash -l
#SBATCH --job-name tf_gpu
#SBATCH --output=/home/adidishe/EightK/out/tf_gpu_%a.out
#SBATCH --error=/home/adidishe/EightK/out/tf_gpu_%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-10

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.autr

# Partition for the job:
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account="punim2039"

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:0:00


module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0
source  ~/venvs/nlp_gpu/bin/activate

python3 train_tf.py ${SLURM_ARRAY_TASK_ID} --cpu=0
