#!/bin/sh
#
#SBATCH --job-name=extract_images
#SBATCH --account=43299_sp0039
#
#SBATCH --partition=Nvidia
#SBATCH --cpus-per-task=1             ## number or regular CPU cores, you can use up to 20 per GPU
#SBATCH --mem=16000
#SBATCH --gres=gpu:tesla:2    ## GPU type and number, alternates: gpu:geforce:2,  gpu:tesla:1, gpu:tesla:2
#SBATCH --time=48:00:00

module purge

module load tensorflow-gpu

# paths
RUNDIR=/storage02/43299_sp0039/burn_multi

INDIR=/storage02/43299_sp0039/burn_multi/Burn_mod


# commands
python ${RUNDIR}/Data.py $INDIR $RUNDIR