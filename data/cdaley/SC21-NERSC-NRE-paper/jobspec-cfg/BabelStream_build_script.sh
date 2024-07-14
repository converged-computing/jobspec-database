#!/bin/bash -l
if [[ "${SLURM_CLUSTER_NAME}" == "escori" ]]; then
    module purge
    module load dgx
    module load nvhpc/21.7
    module load cuda/11.2.1
    module list
    cpus=${SLURM_CPUS_PER_TASK:-32}
    RUN="srun -n 1 -c ${cpus} --cpu-bind=cores"
fi
set -x
set -e

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# make -f OpenMP.make TARGET=NVIDIA NVARCH=cc80 COMPILER=NVIDIA
CXXFLAGS="-Minfo=mp -fast -std=c++11 -DOMP -DOMP_TARGET_GPU -mp=gpu -gpu=cc80"

# make -f CUDA.make NVARCH=sm_80
NVCCFLAGS="-std=c++11 -O3 --use_fast_math -arch=sm_80 -DCUDA"

ARGS="-n 1000" # Repeat 1000 times

# NOTE - The clone is not needed since the repo is added as a submodule.
#if [ ! -d BabelStream ]; then
#    git clone --single-branch --branch main git@github.com:UoB-HPC/BabelStream.git
#fi

cd BabelStream
git checkout 8f9ca7baa77c874897e1691c895729a39d959012
git apply ../add_loop_directive_to_babelstream_8f9ca7.diff

# CUDA
nvcc ${NVCCFLAGS} main.cpp CUDAStream.cu -o cuda-stream
for i in {1..10}; do ${RUN} ./cuda-stream ${ARGS}; done
rm cuda-stream

# OpenMP-4.5
nvc++ ${CXXFLAGS} main.cpp OMPStream.cpp -o omp-stream
for i in {1..10}; do ${RUN} ./omp-stream ${ARGS}; done
rm omp-stream

# OpenMP-5.0 loop
nvc++ ${CXXFLAGS} -DUSE_LOOP main.cpp OMPStream.cpp -o omp-stream
for i in {1..10}; do ${RUN} ./omp-stream ${ARGS}; done
rm omp-stream
