#!/bin/bash
#
#SBATCH --job-name=ssn_workers
#SBATCH --output=ssn_workers.log
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --partition gpu
#SBATCH --nodelist=pgpu03
#SBATCH --array=0-3

CONTAINER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/images/singularity-ssnetserver/singularity-ssnetserver-caffelarbys-cuda8.0.img

WORKDIR=/usr/local/ssnetserver
#WORKDIR=/cluster/kappa/wongjiradlab/twongj01/ssnetserver

# IP ADDRESSES OF BROKER
BROKER=10.246.81.73 # PGPU03
# BROKER=10.X.X.X # ALPHA001

PORT=5560

#GPU_ASSIGNMENTS=/cluster/kappa/wongjiradlab/twongj01/ssnetserver/grid/gpu_assignments.txt
GPU_ASSIGNMENTS=${WORKDIR}/grid/gpu_assignments.txt

module load singularity
singularity exec --nv ${CONTAINER} bash -c "cd ${WORKDIR}/grid && ./run_caffe1worker.sh ${WORKDIR} ${BROKER} ${PORT} ${GPU_ASSIGNMENTS}"
