#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-0:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=demo
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --gres=gpu:rtx8000:1

singularity exec --nv \
--overlay /scratch/xl3136/conda.ext3:ro \
--overlay /scratch/xl3136/dl-sp22-final-project/dataset/labeled.sqsh \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
python barlowtwins.py -n 20 --lr 0.0001
"
