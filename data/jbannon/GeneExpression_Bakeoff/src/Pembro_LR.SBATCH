#!/bin/bash

#SBATCH --job-name=Pembro_LR
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem=20GB
#SBATCH --array=0-5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jjb509@nyu.edu
#SBATCH -o ./slurmouts/Pembro/LR_%j.out

module purge




settings=(STAD.MC STAD.LOO SKCM.MC SKCM.LOO PANCAN.MC PANCAN.LOO)

singularity exec --nv \
            --overlay /scratch/jjb509/GeneExpression_Bakeoff/src/my_overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python permutation_test.py -drug Pembro -model LogisticRegression -settings ${settings[$SLURM_ARRAY_TASK_ID]}"

