#!/bin/bash
#
#SBATCH --job-name=mcskim
#SBATCH --output=log_mcskim.log
#SBATCH --mem-per-cpu=2000
#SBATCH --time=10:00
#SBATCH --partition batch
#SBATCH --array=0-10

CONTAINER=/cluster/tufts/wongjiradlab/larbys/images/singularity-larflow/singularity-larflow-v2.img

WORKDIR_IC=/cluster/kappa/wongjiradlab/twongj01/dllee-ana/skim_mctruth/
INPUTLIST_IC=/cluster/kappa/wongjiradlab/twongj01/dllee-ana/skim_mctruth/rerunlists/mcc9tag2_nueintrinsic_corsika_dlcosmictaggood.list
OUTDIR_IC=/cluster/kappa/wongjiradlab/twongj01/dllee-ana/skim_mctruth/outdir

module load singularity
singularity exec ${CONTAINER} bash -c "cd ${WORKDIR_IC} && ./job.sh $WORKDIR_IC ${INPUTLIST_IC} ${OUTDIR_IC}"
