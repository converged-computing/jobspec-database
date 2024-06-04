#!/bin/bash
#SBATCH --job-name=job_wgpu
#SBATCH --open-mode=append
#SBATCH --output=./slurm/%j_%x.out
#SBATCH --error=./slurm/%j_%x.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH -t 1-23:59
#SBATCH --array=1-9

declare -a METHODS=("icm" "proto" "diayn" "icm_apt" "ind_apt" "aps" "smm" "rnd" "disagreement")


singularity \
  exec --nv \
  --bind /share/apps \
  --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  --overlay /vast/mag1038/icml_dir/urlb.ext3:ro \
  /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
source /ext3/env.sh
conda activate urlb
cd /vast/mag1038/icml_dir/url_benchmark
python pretrain.py domain=point_mass_maze_reach_bottom_right obs_type=pixels agent=${METHODS[$SLURM_ARRAY_TASK_ID]}
"


