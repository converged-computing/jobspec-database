#!/bin/bash
#SBATCH --job-name=nerd_dtu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -c 8
#SBATCH --output="./dtu_bmvs_logs/dtu_bmvs-%a-%j.log"
#SBATCH --open-mode=append
#SBATCH --array="0-22"

scenes=(
split16_dtu_scan122
split16_dtu_scan106
split16_dtu_scan24
split16_dtu_scan83
split16_dtu_scan114
split16_bmvs_stone
split16_dtu_scan40
split16_dtu_scan105
split16_dtu_scan65
split16_dtu_scan69
split16_bmvs_jade
split16_dtu_scan37
split16_dtu_scan97
split16_dtu_scan55
split16_bmvs_clock
split16_dtu_scan110
split16_bmvs_sculpture
split16_bmvs_bear
split16_dtu_scan118
split16_bmvs_dog
split16_bmvs_man
split16_bmvs_durian
split16_dtu_scan63
)

scene="${scenes[$SLURM_ARRAY_TASK_ID]}"

if [[ "$scene" == split16_dtu_* ]]; then
    # DTU scenes
    rwfactor='2'
else
    rwfactor='1'
fi

echo "====== Scene: $scene : $rwfactor ======"

datadir="/export/share/projects/svbrdf/data/dtu_bmvs_nerd/$scene"

python train_nerd.py --datadir "$datadir" --basedir /export/share/projects/svbrdf/data/dtu_bmvs_nerd/nerd --expname "$scene" --config configs/nerd/real_world.txt --rwfactor "$rwfactor"
