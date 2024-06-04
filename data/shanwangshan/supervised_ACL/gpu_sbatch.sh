#!/bin/bash
#SBATCH --job-name=noisy_small_1
#SBATCH --account=project_2003370
#SBATCH -o ./err_out/out_task_number_%A_%a.txt
#SBATCH -e ./err_out/err_task_number_%A_%a.txt
#SBATCH --partition=gpusmall
#SBATCH --time=1:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
##SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:a100:1
#SBATCH --array=1-10
##SBATCH --array=0-1
##SBATCH  --nodelist=r14g06
##module load gcc/8.3.0 cuda/10.1.168
echo $SLURM_ARRAY_TASK_ID
#source activate torch_1.11
module load pytorch/1.11
python train.py -p config/params.yaml -alpha $SLURM_ARRAY_TASK_ID
