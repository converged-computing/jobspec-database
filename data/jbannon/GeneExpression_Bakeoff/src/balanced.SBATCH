#!/bin/bash

#SBATCH --job-name=balanced
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem=20GB
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jjb509@nyu.edu
#SBATCH -o ./slurmouts/balanced_%j.out

module purge


drugs=(Atezo Pembro Nivo)

singularity exec --nv \
            --overlay /scratch/jjb509/GeneExpression_Bakeoff/src/my_overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python model_comparison.py -drug ${drugs[$SLURM_ARRAY_TASK_ID]} -balance 1"
