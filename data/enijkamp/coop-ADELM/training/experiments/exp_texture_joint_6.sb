#!/bin/bash
#SBATCH --job-name="joint_6"
#SBATCH --output="exp_texture_joint_6.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --ntasks-per-node=6
#SBATCH --export=ALL
#SBATCH --gres=gpu:k80:1
#SBATCH -t 24:00:00
#SBATCH -A cla173

#ibrun in verbose mode will give binding detail

module load matlab
matlab -nodisplay -nosplash -nojvm -r "exp_texture_joint_6()"
