#!/bin/bash
#PBS -l select=40:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

# ~~~ EDIT: load your DeepHyper environment here
source /lus/grand/projects/datascience/regele/polaris/deephyper-scalable-bo/build/activate-dhenv.sh

#!!! CONFIGURATION - START
# ~~~ EDIT: used to create the name of the experiment folder
# ~~~ you can use the following variables and pass them to your python script
export problem="jahs"
export search="tpe" # TPE is also valid
export timeout=10200
export SEED=9
#!!! CONFIGURATION - END

export NRANKS_PER_NODE=4
export NDEPTH=$((64 / $NRANKS_PER_NODE))
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH

export DEEPHYPER_LOG_DIR="results/$problem-$search-$NNODES-$SEED"
mkdir -p $DEEPHYPER_LOG_DIR

### Setup Postgresql Database - START ###
export OPTUNA_DB_DIR="$DEEPHYPER_LOG_DIR/optunadb"
export OPTUNA_DB_HOST=$HOST
initdb -D "$OPTUNA_DB_DIR"

# Set authentication policy to "trust" for all users
echo "host    all             all             .hsn.cm.polaris.alcf.anl.gov               trust" >> "$OPTUNA_DB_DIR/pg_hba.conf"

# Set the limit of max connections to 2048
sed -i "s/max_connections = 100/max_connections = 2048/" "$OPTUNA_DB_DIR/postgresql.conf"

# start the server in the background and listen to all interfaces
pg_ctl -D $OPTUNA_DB_DIR -l "$DEEPHYPER_LOG_DIR/db.log" -o "-c listen_addresses='*'" start

createdb hpo
### Setup Postgresql Database - END ###

sleep 5

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    ./set_affinity_gpu_polaris.sh python3 jahs_mpi_solve_tpe.py $SEED

dropdb hpo
pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" stop
rm -rf $OPTUNA_DB_DIR
