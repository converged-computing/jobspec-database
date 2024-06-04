#!/usr/bin/env bash

# Slurm job configuration
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=76
## #SBATCH --gpus-per-task=0
#SBATCH --time=12:00:00
#SBATCH --partition=cpuonly
###SBATCH --partition=accelerated
##SBATCH --gres gpu:1
#SBATCH --account=haicore-project-scc

#SBATCH --job-name=optuna-quartic
#SBATCH --output="/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/paper/quartic/optuna-%j.out"
###SBATCH --job-name=propulate-quartice
###SBATCH --output="/hkfs/work/workspace/scratch/qv2382-propulate/exps/function_benchmark/logs/paper/quartic/propulate-%j.out"

ml purge

export FNAME="quartic"

# pmi2 cray_shasta
SRUN_PARAMS=(
  --mpi="pmi2"
  --label
#  --cpu-bind="ldoms"
)
export FRAMEWORK="optuna"
export EVALS_PER_WORKER=256
rm "/hkfs/work/workspace/scratch/qv2382-bigearthnet/mysqld.sock*"

export DATA_DIR="/hkfs/work/workspace/scratch/qv2382-bigearthnet/"
export BASE_DIR="/hkfs/work/workspace/scratch/qv2382-propulate/"

export SQL_DATA_DIR="${BASE_DIR}sqldata/${FNAME}"
export SQL_CONFIG="${BASE_DIR}exps/function_benchmark/mysqlconfs/${FNAME}.cnf"
export SQL_SOCKET="${BASE_DIR}bigearthnet_kit/mysql/${FNAME}/var/run/mysqld/mysqld.sock"

mkdir "${BASE_DIR}bigearthnet_kit/mysql/${FNAME}/var/run/mysqld/"
rm "${BASE_DIR}bigearthnet_kit/mysql/${FNAME}/var/run/mysqld/*"
chmod 777 "${BASE_DIR}bigearthnet_kit/mysql/${FNAME}"
export SQL_DIR="${BASE_DIR}bigearthnet_kit/mysql/${FNAME}/"

CONTAINER_DIR="${BASE_DIR}containers/"
SINGULARITY_FILE="${CONTAINER_DIR}scratch-tf-sql.sif"
echo "${SINGULARITY_FILE}"

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=100
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

#singularity instance start --bind ${SQL_DATA_DIR}:/var/lib/mysql --bind ${SQL_SOCKET_DIR}:/run/mysqld "${SINGULARITY_FILE}" mysql

srun "${SRUN_PARAMS[@]}" singularity exec \
  --bind /hkfs/work/workspace/scratch/qv2382-propulate/ \
  --bind /hkfs/work/workspace/scratch/qv2382-propulate/bigearthnet_kit/mysql/quartic/var/lib/mysql/:/var/lib/mysql \
  --bind /hkfs/work/workspace/scratch/qv2382-propulate/bigearthnet_kit/mysql/quartic/run/mysqld:/run/mysqld \
  --bind /hkfs/work/workspace/scratch/qv2382-propulate/bigearthnet_kit/mysql/quartic/var/log/mysql/:/var/log/mysql \
  --bind "${DATA_DIR}","/scratch","$TMP" \
  --bind "/hkfs/work/workspace/scratch/qv2382-propulate/propulate/propulate/wrapper.py":"/usr/local/lib/python3.8/dist-packages/propulate/wrapper.py" \
  --bind "/hkfs/work/workspace/scratch/qv2382-propulate/propulate/propulate/propulator.py":"/usr/local/lib/python3.8/dist-packages/propulate/propulator.py" \
  ${SINGULARITY_FILE} \
  bash optuna.sh

