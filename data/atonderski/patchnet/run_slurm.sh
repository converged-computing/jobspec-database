#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --time 72:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /dl_workspaces/%u/patchnet/logs/slurm-%j-run.out
#SBATCH --partition a100
#
singularity exec --bind /dl_workspaces/$USER:/workspace --bind /datasets:/datasets /dl_workspaces/$USER/2021-agp-spatiotemporal/singularity/pytorch21_06.sif bash -c "$@"
#
#EOF
