#!/bin/bash -l
#SBATCH -p ib 
#SBATCH --job-name="UAV-POS-VERIFY"             	 
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-9999
#SBATCH --mail-type=ALL
#SBATCH --mail-user="k.fuger@tuhh.de"
#SBATCH --time=2:00:00
#SBATCH --constraint=OS8
#SBATCH --output=/dev/null

# Execute simulation
pyenv shell 3.11.1
python3 main_hpc.py $SLURM_ARRAY_TASK_ID 0

# Exit job
exit
