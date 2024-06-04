#!/bin/bash 
#SBATCH --job-name=XarmImageReach_norm
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu
#SBATCH --mail-type=END
#SBATCH --array=0
#SBATCH --output=./log/xarm_reach_normgoal/%j_%x_%N.out
#SBATCH --error=./log/xarm_reach_normgoal/%j_%x_%N.err
#SBATCH --export=ALL

srun --mpi=pmi2 -n 1 bash /scratch/$USER/run_singularity_scripts/run_singularity_mpi_torch_2.sh \
     /scratch/$USER/projects/pytorch-visual-learning/config_script.sh