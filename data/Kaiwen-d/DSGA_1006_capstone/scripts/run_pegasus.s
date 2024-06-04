#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-4
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=pegasus
#SBATCH --output=../dataset/outputs/pegasus/shard_%A_%a.out
#SBATCH --error=../dataset/outputs/pegasus/shard_%A_%a.err

module purge

singularity exec $nv \
            --overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:ro \
            /scratch/wz2247/singularity/images/pytorch_22.08-py3.sif  \
            /bin/bash -c "source /ext3/miniconda3/bin/activate; 
            python /scratch/$USER/DSGA_1006_capstone/scripts/pegasus.py $SLURM_ARRAY_TASK_ID"