#!/bin/bash
#SBATCH --job-name=cloth_mass
#SBATCH --partition=psych_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=3-23:00:00
#SBATCH --mail-user=wenyan.bi@yale.edu
#SBATCH --mail-type=ALL
##SBATCH --output=job_%A.log
#SBATCH --output=job_%A_%a.out
#SBATCH --array=0-99%20


pwd; hostname; date
exp_cond='mass'   #stiff|mass
debug=1
SLURM_ARRAY_TASK_ID=0 # debug purpose
SLURM_ARRAY_JOB_ID=0

## =================== [wb] : Create dicts for scenes (e.g., wind -> 1) ====================
scenarios=("wind" "drape" "ball" "rotate")
masses=(0.25 0.5 1.0 2.0 4.0)
stiffnesses=(0.0078125 0.03125 0.125 0.5 2.0)
declare -A scene_idx_dicts

for i in ${!scenarios[@]}; do
  scene_idx_dicts[${scenarios[$i]}]="$(( $i + 1 ))"
done

if [[ debug ]]; then
  echo " "
  echo "============== debug: scene_idx_dicts =============="
  for tmp in "${!scene_idx_dicts[@]}"; do echo "$tmp -> ${scene_idx_dicts[$tmp]}"; done
  echo "===================================================="
fi

## =================== [wb] : Create dicts for possibilities ====================
possibilities=()
for i in ${!scenarios[@]}; do
  for stiffness in "${!stiffnesses[@]}"; do
    for mass in "${!masses[@]}"; do
      # position=$(( $i + 1 ))
      possibilities+=("${scenarios[$i]}_${masses[$mass]}_${stiffnesses[$stiffness]}")
    done
  done
done

## =================== [wb] : Read from csv files ====================
csv_file="cond_file/${exp_cond}/${exp_cond}_${possibilities[$SLURM_ARRAY_TASK_ID]}.csv"


target=($(tail -n +1 ${csv_file} | awk -F ',' '{print $1;}'))
match=($(tail -n +1 ${csv_file} | awk -F ',' '{print $2;}'))
distract=($(tail -n +1 ${csv_file} | awk -F ',' '{print $3;}'))
match_model_idx=($(tail -n +1 ${csv_file} | awk -F ',' '{print $4;}'))
distract_model_idx=($(tail -n +1 ${csv_file} | awk -F ',' '{print $5;}'))
# target_model_idx=($(tail -n +1 ${csv_file} | awk -F ',' '{print $6;}'))
match_mass=($(tail -n +1 ${csv_file} | awk -F ',' '{print $7;}'))
distract_mass=($(tail -n +1 ${csv_file} | awk -F ',' '{print $8;}'))
# target_mass=($(tail -n +1 ${csv_file} | awk -F ',' '{print $9;}'))
match_stiff=($(tail -n +1 ${csv_file} | awk -F ',' '{print $10;}'))
distract_stiff=($(tail -n +1 ${csv_file} | awk -F ',' '{print $11;}'))
# target_stiff=($(tail -n +1 ${csv_file} | awk -F ',' '{print $12;}'))
match_w=($(tail -n +1 ${csv_file} | awk -F ',' '{print $13;}'))
distract_w=($(tail -n +1 ${csv_file} | awk -F ',' '{print $14;}'))
target_w=($(tail -n +1 ${csv_file} | awk -F ',' '{print $15;}'))

total_len=${#target[@]}
total_len=$(( $total_len - 1 ))

#[wb]: Get cur_scene
cur_cloth_scene_mass_bs=${target[1]}
IFS='_' read -r -a array <<< "$cur_cloth_scene_mass_bs"
cur_scene="${array[0]}"


if [[ debug -eq 1 ]]; then
  echo " "
  echo "=============== debug: ========================================================"
  echo "[@] [Cur_exp_cond]         -> ${exp_cond}"
  echo "[@] [Cond_file]            -> ${csv_file}"
  echo "[@] [Cur_infer_cloth]      -> ${scene_idx_dicts[$cur_scene]}/${cur_cloth_scene_mass_bs}"
  echo "[@] -------------------------------------------------"
  echo "==============================================================================="
fi



for i in $(seq 1 $total_len); do
  cur_idx=$(( $i + 0 ))

  if [[ debug -eq 1 ]]; then
    echo " "
    echo "=============== debug: ========================================================"
    echo "[@] [line_num]             -> ${cur_idx}"
    echo "[@] [Match_cloth]          -> ${match[$cur_idx]}"
    echo "[@] [Match_model_idx]      -> ${match_model_idx[$cur_idx]}"
    echo "[@] [Match_mass_prior]     -> ${match_mass[$cur_idx]}"
    echo "[@] [Match_bs_prior]       -> ${match_stiff[$cur_idx]}"
    echo "[@] [Match_w]              -> ${match_w[$cur_idx]}"
    echo "[@] -------------------------------------------------"
    echo "[@] [Distract_cloth]       -> ${distract[$cur_idx]}"
    echo "[@] [Distract_model_idx]   -> ${distract_model_idx[$cur_idx]}"
    echo "[@] [Distract_mass_prior]  -> ${distract_mass[$cur_idx]}" 
    echo "[@] [Distract_bs_prior]    -> ${distract_stiff[$cur_idx]}"
    echo "[@] [Distract_w]           -> ${distract_w[$cur_idx]}"
    echo "==============================================================================="
  fi



  ./run.sh julia src/exp_basic1.jl ${scene_idx_dicts[$cur_scene]}/${cur_cloth_scene_mass_bs} \
                                  ${exp_cond} \
                                  ${cur_idx}_${match[$cur_idx]}_${match_model_idx[$cur_idx]} \
                                  ${match_mass[$cur_idx]} \
                                  ${match_stiff[$cur_idx]} \
                                  ${match_w[$cur_idx]} \
                                  ${SLURM_ARRAY_JOB_ID}


  ./run.sh julia src/exp_basic1.jl ${scene_idx_dicts[$cur_scene]}/${cur_cloth_scene_mass_bs} \
                                  ${exp_cond} \
                                  ${cur_idx}_${distract[$cur_idx]}_${distract_model_idx[$cur_idx]} \
                                  ${distract_mass[$cur_idx]} \
                                  ${distract_stiff[$cur_idx]} \
                                  ${distract_w[$cur_idx]} \
                                  ${SLURM_ARRAY_JOB_ID}

done
date





# echo ${total_len}
# for i in $(seq 1 $total_len); do
# echo ${match_stiff[1]}
