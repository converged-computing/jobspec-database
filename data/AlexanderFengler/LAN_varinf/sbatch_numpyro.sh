#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J numpyro_sampler

# priority
##SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output slurm/numpyro_sampler_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=32:00:00
#SBATCH --mem=8G
##SBATCH -c 4
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-19

# --------------------------------------------------------------------------------------

# Setup
source /users/afengler/.bashrc
module load cudnn/8.1.0
module load cuda/11.1.1
#module load cuda/11.3.1
module load gcc/10.2

conda deactivate
conda deactivate
conda activate pymc-gpu

# Read in arguments:
model=ddm
modeltype=singlesubject
nwarmup=100
nmcmc=100
nchains=2
idrange=10
machine=cpu
progressbar=0

nvidia-smi
lscpu

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --model | -m)
                echo "passing model $2"
                model=$2
                ;;
            --modeltype | -t)
                echo "passing modeltype $2"
                modeltype=$2
                ;;
            --machine | -i)
                echo "passing modeltype $2"
                machine=$2
                ;;                
            --nwarmup | -w)
                echo "passing nwarmup $2"
                nwarmup=$2
                ;;
            --nmcmc | -c)
                echo "passing nmcmc $2"
                nmcmc=$2
                ;;
            --nchains | -n)
                echo "passing nmcmc $2"
                nchains=$2
                ;;
            --progressbar | -b)
                echo "passing nmcmc $2"
                progressbar=$2
                ;;
            --idrange | -i)
                echo "passing idrange $2"
                idrange=$2
        esac
        shift 2
    done

python -u run_inference_numpyro.py --model $model \
                                   --modeltype $modeltype \
                                   --machine $machine \
                                   --nchains $nchains \
                                   --nwarmup $nwarmup \
                                   --nmcmc $nmcmc \
                                   --progressbar $progressbar \
                                   --idmin $((SLURM_ARRAY_TASK_ID*idrange)) \
                                   --idmax $((SLURM_ARRAY_TASK_ID*idrange + idrange))