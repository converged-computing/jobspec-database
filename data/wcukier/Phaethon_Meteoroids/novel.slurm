#!/bin/bash
#SBATCH --job-name=novel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=300M
#SBATCH	--time=03:02:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --mail-user=****
#SBATCH --array=0-99

module purge
module load anaconda3/2021.5

python main.py $SLURM_ARRAY_TASK_ID 0 100 2000
