#!/bin/bash -l
#PBS -l nodes=1:ppn=36
#PBS -l pmem=5GB
#PBS -l walltime=12:00:00
#PBS -A lp_louvre_esat_students
#PBS -m ae -M vasileios.lemonidis@student.kuleuven.be

module purge
module load matlab/R2019a

if [[ ! -z "$PBS_O_WORKDIR" ]] ; then
cd $PBS_O_WORKDIR
fi
#ENV TO SET:(DEFAULTS)
############
#DATA_ROOT: the directory of the DATA (../SAMPLE_DATA/)
#DATASET_INDEX: dataset to use, 1 or 2 (1)
#RESULTS_ROOT: the directory of the results (../results/)
#THREADS: Number of threads to use (max set by local)
#CHROMOSOME: Chomosome to analyze (1:22)
#BLOCK_SIZE: Block Size for CCA (2000)
#MEDIAN_IMPUTE: Whether to perform median imputation,1 (very fast), or not, 0 (very slow) (0)
#PHENO_PATH: Whether to use a specific path for the mat file of the phenotype, other than the default one.
############

MAIN_DIR=vlThesis

export DATA_ROOT=$VSC_DATA/$MAIN_DIR/SAMPLE_DATA/
export DATASET_INDEX=1 # Use discover dataset
export IMPUTE_STRATEGY=no
export RESULTS_ROOT=$VSC_DATA/$MAIN_DIR/results/
export SCRATCH_ROOT=$VSC_SCRATCH/$MAIN_DIR/
if [[ ! -f $SCRATCH_ROOT ]]; then
mkdir -p $SCRATCH_ROOT
fi
export CHROMOSOME=${PBS_ARRAYID}


### The following is written in the hpc site, I am following it blindly.
# use temporary directory (not $HOME) for (mostly useless) MATLAB log files
# subdir in $TMPDIR (if defined, or /tmp otherwise)
export MATLAB_LOG_DIR=$(mktemp -d -p ${TMPDIR:-/tmp})

# configure MATLAB Compiler Runtime cache location & size (1GB)
# use a temporary directory in /dev/shm (i.e. in memory) for performance reasons
export MCR_CACHE_ROOT=$(mktemp -d -p /dev/shm)
export MCR_CACHE_SIZE=1024MB
####

installation="$(dirname $(dirname $(which matlab)))"
ctime=`date +"%Y_%m_%d_%T"`
mkdir -p logs
./run_demoBrainAsymmetryGenome.sh $installation > "logs/genome${PBS_ARRAYID}${ctime}.log"

