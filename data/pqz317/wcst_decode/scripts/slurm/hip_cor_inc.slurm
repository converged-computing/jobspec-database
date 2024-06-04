#!/bin/bash
#SBATCH --job-name=hip_cor_inc

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

array=("cor" "inc")

cat ${array[$SLURM_ARRAY_TASK_ID]}

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231103_decode_features_subpop_abstract.py \
    --subtrials_path /data/${array[$SLURM_ARRAY_TASK_ID]}_bal_trials.pickle --subtrials_name ${array[$SLURM_ARRAY_TASK_ID]} \
    --subpop_path /data/hip_subpop.pickle --subpop_name hip