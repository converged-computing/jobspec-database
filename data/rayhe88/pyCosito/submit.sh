#!/bin/bash
#PBS -V
#PBS -l select=8:system=sunspot
#PBS -l place=scatter
#PBS -l walltime=2:00:00
#PBS -q workq
#PBS -A catalysis_aesp_CNDA
#PBS -N Fireworks
#PBS -l filesystems=home:gila


module load cray-libpals/1.2.12 cray-pals/1.2.12
module load append-deps/default
module load prepend-deps/default
module load libfabric/1.15.2.0
module load oneapi/eng-compiler/2023.05.15.003
module load intel_compute_runtime/release/agama-devel-627
module load mpich/52.2/icc-all-pmix-gpu
module load spack cmake

source $IDPROOT/etc/profile.d/conda.sh
conda activate /lus/gila/projects/catalysis_aesp_CNDA/raymundohe/pynta_env

export NUMEXPR_MAX_THREADS=256


module list

export EXE=/lus/gila/projects/catalysis_aesp_CNDA/raymundohe/PWDFT/build_sycl/pwdft
export GPU=/lus/gila/projects/catalysis_aesp_CNDA/raymundohe/bin/gpu_tile_compact.sh

cd $PBS_O_WORKDIR
echo " ===================================  TEST  ===================================="
echo "          JobId  : $PBS_JOBID"
echo "Running on host  : `hostname`"
echo "Running on nodes : `cat $PBS_NODEFILE`"
echo "numactl -H"
numactl -H
echo " ===================================  TEST  ===================================="

unset EnableWalkerPartition
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

cd $PBS_O_WORKDIR

echo -n " Starting  :"
date

python run.py
echo " ===================================  DONE  ===================================="
echo -n " Finishing :"
date
