#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --array=1
#SBATCH --mem=60GB
#SBATCH --job-name=r4train
#SBATCH --output=/scratch/cg3306/climate/subgrid/gz21/slurm/echo/r4train_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/subgrid/gz21/slurm/echo/r4train_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
echo "$(date)"
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/subgrid/gz21/overlay-15GB-500K.ext3:ro\
	 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "
		source /ext3/env.sh;
		mlflow run -e four-regions-train . --env-manager local --experiment-name r4train --run-name full;
	"
echo "$(date)"