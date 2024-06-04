#!/bin/bash

#SBATCH --job-name=visuals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=aeb783@nyu.edu
#SBATCH --output=slurm_%j.out


singularity exec \
	--overlay /scratch/aeb783/pangeo/pytorch1.7.0-cuda11.0.ext3:ro \
	/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	bash -c "source /ext3/env.sh;
python -u /home/aeb783/mom6/MOM6-examples/scripts/MOM_ss_visuals.py"
