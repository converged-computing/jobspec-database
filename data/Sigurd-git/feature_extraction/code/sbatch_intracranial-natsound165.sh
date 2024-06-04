#!/bin/bash
#SBATCH -p doppelbock -t 04:00:00
#SBATCH -c 8
#SBATCH -a 0-8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1 -x bhg0044,bhg0046,bhg0047,bhg0048
#SBATCH -o /scratch/snormanh_lab/shared/projects/intracranial-natsound165/analysis/logs/%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guoyang_liao@urmc.rochester.edu

project='intracranial-natsound165'

non_list=(modulus real rect)
modulation_list=(spectempmod specmod tempmod)
length_modulation_list=${#modulation_list[@]}
length_non_list=${#non_list[@]}

# if $SLURM_ARRAY_TASK_ID is smaller than length_modulation_list * length_non_list, then we are computing spectrotemporal models
if [ $SLURM_ARRAY_TASK_ID -lt $((length_modulation_list * length_non_list)) ]; then
    modulation_index=$(( SLURM_ARRAY_TASK_ID % length_modulation_list))
    non_list_index=$(( SLURM_ARRAY_TASK_ID / length_modulation_list))
    modulation_type=${modulation_list[modulation_index]}
    non_lin=${non_list[non_list_index]}
    model=cochleagram_spectrotemporal
    python -u /scratch/snormanh_lab/shared/code/snormanh_lab_python/feature_extraction/code/main.py "env=$project" "env.feature=$model" "env.spectrotemporal.modulation_type=$modulation_type" "env.spectrotemporal.nonlin=$non_lin" "env.device=cpu" "$@"
else
    index=$((SLURM_ARRAY_TASK_ID - length_modulation_list * length_non_list))
    model_list=("ast" "cochdnn" "cochresnet" "hubert" "spectrogram")
    model=${model_list[$index]}
    python -u /scratch/snormanh_lab/shared/code/snormanh_lab_python/feature_extraction/code/main.py "env=$project" "env.feature=$model" "$@"
fi
