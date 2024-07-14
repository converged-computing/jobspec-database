#!/bin/bash
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

KOKKOS_PATH=$(pwd)/kokkos

# NOTE - The clone is not needed since the repo is added as a submodule.
#    if [ ! -d kokkos ]; then
#        git clone --single-branch --branch develop https://github.com/kokkos/kokkos.git
#    fi 
cd kokkos
git checkout 0b61c81a6756610a3e8edb2061dbe7b882a054e8

if [ -d build_cuda_nvhpc ]; then
    rm -rf build_cuda_nvhpc
fi
mkdir build_cuda_nvhpc && cd build_cuda_nvhpc
cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=${KOKKOS_PATH}/install_cuda_nvhpc \
  -D CMAKE_CXX_COMPILER=${KOKKOS_PATH}/bin/nvcc_wrapper \
  -D CMAKE_C_COMPILER=gcc \
  -D CMAKE_CXX_STANDARD=17 \
  -D Kokkos_ARCH_AMPERE80=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
  ..
make -j8
make install
cd ..

if [ -d build_ompt_nvhpc ]; then
    rm -rf build_ompt_nvhpc
fi
mkdir build_ompt_nvhpc && cd build_ompt_nvhpc
cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=${KOKKOS_PATH}/install_ompt_nvhpc \
  -D CMAKE_CXX_COMPILER=nvc++ \
  -D CMAKE_CXX_STANDARD=17 \
  -D Kokkos_ARCH_AMPERE80=ON \
  -D Kokkos_ENABLE_OPENMPTARGET=ON \
  -D Kokkos_ENABLE_TESTS=ON \
  -D Kokkos_ENABLE_IMPL_DESUL_ATOMICS=OFF \
  -D CMAKE_CXX_FLAGS="-mp=gpu -gpu=cc80" \
  ..
make -j8
make install
${RUN} ./core/unit_test/KokkosCore_IncrementalTest_OPENMPTARGET
cd ..

cd ..
