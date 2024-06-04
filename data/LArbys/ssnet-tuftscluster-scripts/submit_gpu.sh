#!/bin/bash
#
#SBATCH --job-name=ssnet
#SBATCH --output=log_ssnet.txt
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --nodelist=pgpu01

#CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-dllee-ubuntu/singularity-dllee-ssnet-nvidia375.39-cpuonly.img
CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-dllee-ssnet/singularity-dllee-ssnet-nvidia384.66.img
WORKDIR=/cluster/kappa/90-days-archive/wongjiradlab/grid_jobs/ssnet-tuftscluster-scripts
INPUTLISTDIR=${WORKDIR}/inputlists
JOBLIST=${WORKDIR}/rerunlist.txt

OUTDIR=/cluster/kappa/90-days-archive/wongjiradlab/larbys/data/comparison_samples/extbnb_wprecuts_reprocess/out_week10132017/ssnet_p02

module load singularity

#singularity exec --nv ${CONTAINER} bash -c "cd ${WORKDIR} && source run_tuftsgrid_ssnet.sh ${WORKDIR} ${INPUTLISTDIR} ${OUTDIR} ${JOBLIST}"
python manage_tufts_gpu_jobs.py ${CONTAINER} ${WORKDIR}
