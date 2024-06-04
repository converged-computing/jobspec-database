#!/bin/bash

#SBATCH -o logs/output.train.%j
#SBATCH -e logs/error.train.%j
#SBATCH -D ./
#SBATCH -J train
#SBATCH -p gpu.q
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task
#SBATCH --cpus-per-task=1  # assign one core to that first task
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=50GB
#SBATCH --export=ALL

module load anaconda/3/2021.11
source activate /u/atya/conda-envs/tf-gpu4
#combined_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
#time srun python make_uniformz.py $combined_id $SLURM_ARRAY_TASK_ID
#time srun python single_index_predict.py $SLURM_JOB_ID attention > logs/output.train.$SLURM_JOB_ID
#time srun python chunk_classifier.py $SLURM_JOB_ID cnn > logs/output.train.$SLURM_JOB_ID
#time srun python single_index_predict_resume.py $SLURM_JOB_ID attention 55 13500 1231538_55 > logs/output.train.$SLURM_JOB_ID
#time srun python single_index_predict_f.py --job_id $SLURM_JOB_ID --model_type cnn --trial_index 203 --snr_group -1 --max_seconds 13500 > logs/output.train.$SLURM_JOB_ID
#time srun python single_index_predict_z.py --job_id $SLURM_JOB_ID --model_type attention --trial_index 853 --snr_group 1 --max_seconds 13500 --previous_job_id 2548494 > logs/output.train.$SLURM_JOB_ID
#time srun python single_index_predict_snr.py $SLURM_JOB_ID cnn runBB 985 13500 > logs/output.train.$SLURM_JOB_ID
time srun python single_index_predict_z_with_pvol.py --job_id $SLURM_JOB_ID --model_type attention --trial_index 853 --snr_group 1 --max_seconds 71500 > logs/output.train.$SLURM_JOB_ID


#time srun python create_ppdot_all.py 0 3000

#SBATCH --mail-type=END,FAIL --mail-user=s6abtyag@uni-bonn.de
#SBATCH --array=13-30
#SBATCH --dependency=afterok:1585071
# --previous_job_id 1280525 cnn index 985 high snr case runBB
# --previous_job_id 1231538_55 attention --trial_index 55 case runBB
# --previous_job_id 2548494 attention --trial_index 853 case runBD
# trail index 102 for runBC




