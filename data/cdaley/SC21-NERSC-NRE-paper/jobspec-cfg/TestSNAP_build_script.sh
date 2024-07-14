#!/bin/bash -l
if [[ "${SLURM_CLUSTER_NAME}" == "escori" ]]; then
    module purge
    module load dgx
    module load nvhpc/21.3
    module load cuda/11.0.2
    module load cmake/3.18.2
    module load gcc/8.3.0
    module list
    cpus=${SLURM_CPUS_PER_TASK:-32}
    RUN="srun -n 1 -c ${cpus} --cpu-bind=cores"
fi
set -x
set -e

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

START=$(pwd)
KOKKOS_PATH=$(pwd)/kokkos
cd ${KOKKOS_PATH}

if [ ! -d build_cuda_nvhpc ]; then
    echo "Cuda build of kokkos not found. Please run the Kokkos script first."
    exit 1
fi

if [ ! -d build_ompt_nvhpc ]; then
    echo "OpenMPTarget build of kokkos not found. Please run the Kokkos script first."
    exit 1
fi
cd ${START}

# NOTE - The clone is not needed since the repo is added as a submodule.
#if [ ! -d TestSNAP ]; then
#    git clone --single-branch --branch Kokkos-nvhpc https://github.com/rgayatri23/TestSNAP.git TestSNAP-Kokkos
#fi

cd TestSNAP
git checkout 29cfb02164152eb1c4a3e5bf55f679deb04e9205

if [ -d build_cuda_nvhpc ]; then
    rm -rf build_cuda_nvhpc
fi
mkdir build_cuda_nvhpc && cd build_cuda_nvhpc
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_COMPILER=${KOKKOS_PATH}/install_cuda_nvhpc/bin/nvcc_wrapper \
      -D CMAKE_C_COMPILER=$(which gcc) \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_EXTENSIONS=OFF \
      -D Kokkos_ROOT=${KOKKOS_PATH}/install_cuda_nvhpc \
      -D ref_data=14 \
      ..
make
for i in {1..10}; do ${RUN} ./test_snap -ns 100; done
cd ..

if [ -d build_ompt_nvhpc ]; then
    rm -rf build_ompt_nvhpc
fi
mkdir build_ompt_nvhpc && cd build_ompt_nvhpc
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_COMPILER=nvc++ \
      -D CMAKE_C_COMPILER=nvc \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CXX_EXTENSIONS=OFF \
      -D Kokkos_ROOT=${KOKKOS_PATH}/install_ompt_nvhpc \
      -D ref_data=14 \
      -D CMAKE_CXX_FLAGS="-mp=gpu -gpu=cc80" \
      ..
make
for i in {1..10}; do ${RUN} ./test_snap -ns 100; done
cd ..

cd ..
