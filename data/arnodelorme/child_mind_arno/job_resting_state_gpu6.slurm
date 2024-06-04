#!/bin/bash
#SBATCH --job-name="ChildMindDL"
#SBATCH --output="Child_mind_p100.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --mem=25G
#SBATCH --export=ALL
#SBATCH -t 48:00:00
#SBATCH --mail-user=adelorme@ucsd.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A TG-IBN140002
source ~/.bashrc
cd /projects/ps-nemar/child_mind_2020
module load matlab
matlab -nodisplay -nosplash -nodesktop < /projects/ps-nemar/child_mind_2020/restingstate_dl_comet6.m
