#!/bin/bash
#
#SBATCH --job-name=training_ubresnet
#SBATCH --output=log_training_ubresnet.log
#SBATCH --mem-per-cpu=2500
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03
#SBATCH --array=0-5

CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-larbys-pytorch/singularity-larbys-pytorch-0.3-larcv1-nvidia384.66.img
WORKDIR_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/ubresnet/training/workdir
REPODIR_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/ubresnet
RUNSCRIPT_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/ubresnet/training/grid_scripts/larcv1_run_training.sh
TRAININGSCRIPT_IN_CONTAINER=/cluster/kappa/wongjiradlab/twongj01/ubresnet/training/grid_scripts/train_ubresnet_wlarcv1_tuftsgrid.py

RUNSCRIPT_ARGS="${WORKDIR_IN_CONTAINER} ${TRAININGSCRIPT_IN_CONTAINER}"

module load singularity
singularity exec --nv ${CONTAINER} bash -c "mkdir -p ${WORKDIR_IN_CONTAINER} && cd ${REPO_IN_CONTAINER} && source ${RUNSCRIPT_IN_CONTAINER} ${REPODIR_IN_CONTAINER} ${RUNSCRIPT_ARGS}"