#!/bin/bash -l
#SBATCH --account=m1302
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#--------------------------------
##SBATCH --cpus-per-task=32
##SBATCH --constraint=cpu
#--------------------------------
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu
#--------------------------------
#
module reset
#module unload darshan Nsight-Compute Nsight-Systems cudatoolkit craype-accel-nvidia80 gpu
module unload darshan
module load Nsight-Compute Nsight-Systems cudatoolkit craype-accel-nvidia80 gpu
module load python PrgEnv-gnu cray-fftw cray-hdf5-parallel cray-libsci
module load libfabric


#-------------------------------------------------------------------------------
# get a unique tag
TAG=`date '+%Y-%m-%d-%H%M'`-`uuidgen -t | head -c 4`
# get the dir list
DBS_DIR=${HOME}/lib-PMI-4.1.2
MPI_DIR=${DBS_DIR}
HOME_DIR=${HOME}/rmem
SCRATCH_DIR=/pscratch/sd/t/tgillis/rmem_${TAG}_${SLURM_JOBID}

#-------------------------------------------------------------------------------
# asan with cuda, from https://github.com/google/sanitizers/issues/629
# asan with pthread, from https://github.com/google/sanitizers/issues/1171
export ASAN_OPTIONS=protect_shadow_gap=0:use_sigaltstack=0
#-------------------------------------------------------------------------------
export FI_HMEM_CUDA_USE_GDRCOPY=1
#export FI_CXI_RDZV_THRESHOLD=4096
echo "--------------------------------------------------"
echo "running in ${SCRATCH_DIR}"
echo "FI_CXI_RDZV_EAGER_SIZE = ${FI_CXI_RDZV_EAGER_SIZE}"
echo "FI_CXI_RDZV_THRESHOLD = ${FI_CXI_RDZV_THRESHOLD}"
echo "FI_CXI_RDZV_GET_MIN = ${FI_CXI_RDZV_GET_MIN}"
echo "--------------------------------------------------"

#-------------------------------------------------------------------------------
mkdir -p ${SCRATCH_DIR}
mkdir -p ${SCRATCH_DIR}/build
cd ${SCRATCH_DIR}
cp -r ${HOME_DIR}/src .
cp -r ${HOME_DIR}/make_arch .
cp -r ${HOME_DIR}/Makefile .


#-------------------------------------------------------------------------------
# need to put 2 here to let some space of the thread!
MPI_OPT="-n 2 -ppn 1 -l --bind-to core:2"
#-------------------------------------------------------------------------------
export PMI_DIR=${DBS_DIR}

for device in 0 1; do
    export USE_CUDA=${device}
    make clean
    make info debug
    ldd rmem
    #test delivery
    declare -a test=(
        "-r am -d am -c delivery"
        "-r am -d am -c fence"
        #"-r am -d am -c counter"
        #"-r am -d am -c cq_data"
        #"-r tag -d tag -c cq_data"
    )
    for RMEM_OPT in "${test[@]}"; do
        echo "==> ${MPI_OPT} with ${RMEM_OPT} - CUDA? ${USE_CUDA}"
        FI_PROVIDER="cxi" ${MPI_DIR}/bin/mpiexec ${MPI_OPT} ./rmem ${RMEM_OPT}
        #FI_PROVIDER="psm3" ${MPI_DIR}/bin/mpiexec ${MPI_OPT} ./rmem ${RMEM_OPT}
    done
done

##-------------------------------------------------------------------------------
#FI_PROVIDER="psm3" ${MPI_DIR}/bin/mpiexec -ppn 1 -l --bind-to core \
#    -n 1 perf record --call-graph dwarf -o perf.0 ./rmem : \
#    -n 1 perf record --call-graph dwarf -o perf.1 ./rmem
#
#
#for id in 0 1
#do
#    perf script -i perf.${id} > out.${id}.perf
#    ${HOME}/FlameGraph/stackcollapse-perf.pl out.${id}.perf > out.${id}.folded
#    ${HOME}/FlameGraph/flamegraph.pl out.${id}.folded > ${id}.svg
#done
#
#
##-------------------------------------------------------------------------------
#mkdir -p ${SCRATCH_DIR}
#mkdir -p ${SCRATCH_DIR}/build
#cd ${SCRATCH_DIR}
#cp -r ${HOME_DIR}/src .
#cp -r ${HOME_DIR}/make_arch .
#cp -r ${HOME_DIR}/Makefile .
#
## asan with cuda, from https://github.com/google/sanitizers/issues/629
## asan with pthread, from https://github.com/google/sanitizers/issues/1171
#export ASAN_OPTIONS=protect_shadow_gap=0:use_sigaltstack=0
##export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
#export FI_CXI_OPTIMIZED_MRS=false
##-------------------------------------------------------------------------------
##---------- NO GPU
##echo "---------------------------- CXI"
##make clean
##USE_CUDA=0 OPTS="-DNO_WRITE_DATA" make fast
##FI_PROVIDER="cxi" \
##${DBS_DIR}/bin/mpiexec -n 2 -ppn 1 -l --bind-to core:2 ./rmem
#echo "---------------------------- CXI"
#make clean
#USE_CUDA=1 OPTS="-DNO_WRITE_DATA" make fast
#FI_PROVIDER="cxi" \
#${DBS_DIR}/bin/mpiexec -n 2 -ppn 1 -l --bind-to core:2 ./rmem
#
#
#end of life
