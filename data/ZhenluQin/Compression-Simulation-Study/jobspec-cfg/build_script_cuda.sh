#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building the example

set -x
set -e

ml cmake cuda/11.4 adios2 libfabric gcc/9

mgard_src_dir=/ccs/home/jieyang/MGARD
mgard_install_dir=${mgard_src_dir}/install-cuda-summit
#nvcomp_install_dir=${mgard_src_dir}/external-cuda-summit/nvcomp/install
#zstd_install_dir=${mgard_src_dir}/external-cuda-summit/zstd/install
#protobuf_install_dir=${mgard_src_dir}/external-cuda-summit/protobuf/install

export LD_LIBRARY_PATH=${mgard_install_dir}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${mgard_install_dir}/lib64:${LD_LIBRARY_PATH}


rm -rf build_cuda
mkdir -p build_cuda 
cmake -S .  -B ./build_cuda \
	    -DCMAKE_MODULE_PATH=${mgard_src_dir}/cmake\
	    -Dmgard_ROOT=${mgard_src_dir}/install-cuda-summit\
	    -DCMAKE_PREFIX_PATH="${mgard_install_dir}"

cd build_cuda && make && cd ..

# cmake --build ./build
