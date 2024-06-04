#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

#########################################
#Positional Parameters
# $1 is the name of algorithm
# $2 is the name of environment
# $3 is the set of quantization bit(QAT), using 32 bit means no QAT
# $4 is the test times(seeds number)
# $5 is timestep for training
########################################

set -e

SLURM_ARRAY_TASK_ID=204
Learning_rate=(0.0005 0.001 0.005 0.01 0.05 0.1)
Rho=(0.01 0.02 0.05 0.1)
lambda_hero=(0.05 0.1 0.5 1.0 5.0)
agents=("a2c" "sac")
envs=("Pendulum-v1")
seeds=(1 2 3 4)

remainder=$SLURM_ARRAY_TASK_ID
base=$((${#Rho[@]}*${#lambda_hero[@]}*${#agents[@]}*${#envs[@]}*${#seeds[@]}))
lr=${Learning_rate[$(($remainder/$base))]}

echo "Learning rate: $lr"

remainder=$(($remainder%$base))
base=$((${#lambda_hero[@]}*${#agents[@]}*${#envs[@]}*${#seeds[@]}))
rho=${Rho[$(($remainder/$base))]}

echo "Rho: $rho"

remainder=$(($remainder%$base))
base=$((${#agents[@]}*${#envs[@]}*${#seeds[@]}))
lambda=${lambda_hero[$(($remainder/$base))]}

echo "Lambda: $lambda"

remainder=$(($remainder%$base))
base=$((${#envs[@]}*${#seeds[@]}))
agent=${agents[$(($remainder/$base))]}

echo "agent: $agent"

remainder=$(($remainder%$base))
base=$((${#seeds[@]}))
env=${envs[$(($remainder/$base))]}

echo "env: $env"
#
#remainder=$(($remainder%$base))
#base=$seeds
#opt=${Optimizer[$(($remainder/$base))]}

#echo "Optimizer: $opt"
#
i=$(($remainder%$base))
((i++))
echo "Seed Number: $i"

python train.py --algo $agent --env $env --device cuda --optimize-choice HERO --quantize 32 -P --rho $rho -params learning_rate:$lr --track -n 1000000 --wandb-entity "qorl"
ptq_all.sh $agent $env "logs/$agent/"$env"_32bit_"$HERO"_$i" HERO $i $rho $lr