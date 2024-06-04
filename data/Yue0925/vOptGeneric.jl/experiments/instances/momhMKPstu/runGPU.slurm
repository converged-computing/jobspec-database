#!/bin/bash
#SBATCH --job-name=basic_algo
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=./basic_algo.out
#SBATCH --error=./basic_algo.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=yue.zhang@lipn.univ-paris13.fr

source ~/.bashrc

for file in ./MOBKP/set3/*; do
    echo "$file"
    julia vOptMomkp.jl "$file"
done

# julia vOptMomkp.jl ./MOBKP/set3/Wcollage-tube.DAT
