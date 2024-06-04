#!/bin/bash 
#SBATCH --job-name=side_XarmPickandPlace
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --array=0
#SBATCH --output=./log/xarm_pickandplace_fixed_normgoal/%j_%x_%N.out
#SBATCH --error=./log/xarm_pickandplace_fixed_normgoal/%j_%x_%N.err
#SBATCH --export=ALL

srun --mpi=pmi2 -n 1 bash /scratch/$USER/run_singularity_scripts/run_singularity_mpi_torch_3.sh \
     /scratch/$USER/projects/pytorch-visual-learning/config_script_pickandplace.sh