#!/bin/bash
#SBATCH -A stf006
#SBATCH -J test_SSL
#SBATCH -o "testjob_%j"
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -t 2:00:00
#SBATCH --export NONE

unset SLURM_EXPORT_ENV
cd $SLURM_SUBMIT_DIR

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

module load cray-python
module load PrgEnv-gnu 
module load amd-mixed/5.4.3 
module load craype-accel-amd-gfx90a

source $PROJWORK/stf006/muraligm/software/miniconda3-frontier/bin/activate
conda activate /lustre/orion/stf006/proj-shared/muraligm/ML/SSL_ALPNet/SSL_ALPNet_frontier39

echo "===============STARTING TIME==============="
date

echo "***************Trying without srun***************"
./examples/train_ssl_abdominal_mri.sh

# echo "***************Trying srun***************"
# OMP_NUM_THREADS=1 srun -u --gpus-per-task=1 --gpu-bind=closest -N1 -n1 -c1 ./examples/train_ssl_abdominal_mri.sh

wait

echo "===============ENDING TIME==============="
date

