#!/bin/bash
#SBATCH --job-name=muscle_MSA
#SBATCH --output=msa_%j.out
#SBATCH --error=msa_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G

module load mamba intel
source activate main_env

A3M_FILE=$1
MODE=$2

python /home/adaddi/data/muscle5/muscle_MSA.py --a3m_file "$A3M_FILE" --mode "$MODE"
