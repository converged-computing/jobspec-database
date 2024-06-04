#!/bin/bash
#SBATCH --time=32:00:00
#SBATCH --array=2-3
#SBATCH --mem=60GB
#SBATCH --job-name=gitrain
#SBATCH --output=/scratch/cg3306/climate/subgrid/gz21/slurm/echo/gitrain_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/subgrid/gz21/slurm/echo/gitrain_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
echo "$(date)"
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/subgrid/gz21/overlay-15GB-500K.ext3:ro\
	 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "
		source /ext3/env.sh;
		mlflow run -e global-interior-train . --env-manager local --experiment-name gitrain --run-name full;
	"
echo "$(date)"