#!/bin/bash
#SBATCH --job-name=liquid_gen
#SBATCH --partition=psych_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-user=yuting.zhang@yale.edu
#SBATCH --mail-type=ALL
##SBATCH --output=job_%A.log
#SBATCH --output=job_%A_%a.out

pwd; hostname; date

./run.sh julia src/exp_basic.jl 2/boxwithahole_16
#./run.sh julia src/exp_basic.jl 3/oneobject_104

date
if [[ "$@" =~ "on" ]];then
    rm -rf ../SPlisHSPlasH/bin/output/simulation_422
fi


