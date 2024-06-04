#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:15:00
#PBS -q workq
#PBS -A candle_aesp_CNDA
#PBS -m abe
#PBS -M avasan@anl.gov

OUTPUT_DIR=/lus/gila/projects/candle_aesp_CNDA/avasan/DockingSurrogates/Inference/Inference_Scaling/ST_Sort_CheckList/logs
WORKDIR=/lus/gila/projects/candle_aesp_CNDA/avasan/DockingSurrogates/Inference/Inference_Scaling/ST_Sort_CheckList/
cd ${WORKDIR}
DATA_FORMAT="channels_last"

PRECISION="float32"
 
# Adjust the local batch size:
LOCAL_BATCH_SIZE=1
 
#####################################################################
# FRAMEWORK Variables that make a performance difference for tf:
#####################################################################
 
# Toggle tf32 on (or don't):
ITEX_FP32_MATH_MODE=TF32
 
# For cosmic tagger, this improves performance:
# (for reference, the default is "setenv ITEX_LAYOUT_OPT \"1\" ")
unset ITEX_LAYOUT_OPT
 
# Set some CCL backend variables:
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ALLREDUCE=topo
export CCL_LOG_LEVEL=warn
export NUMEXPR_MAX_THREADS=208

#####################################################################
# End of perf-adjustment section
#####################################################################
 
 
#####################################################################
# Environment set up, using the latest frameworks drop
#####################################################################
 
# Frameworks have a different oneapi backend at the moment:

module load frameworks/2023.05.15.001

# activate python environment
source $IDPROOT/bin/activate
conda activate /lus/gila/projects/candle_aesp_CNDA/avasan/envs/conda_tf_env/

#module restore
module list

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
export RANKS_PER_NODE=12           # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
export PROCS_PER_TILE=1

echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"

export ZEX_NUMBER_OF_CCS=0:1,1:1,2:1,3:1,4:1,5:1
mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} ./set_ze_mask_multiinstance.sh python smiles_regress_transformer_run_large.py
