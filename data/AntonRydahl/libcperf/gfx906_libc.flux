#!/bin/bash
set -o nounset

# Job settings
#flux: --job-name=gfx906_libc
# Request one node
#flux: --nodes=1
# Run one process per node
#flux: -n 1
# Request 32 cores per node
#flux: -c 32
# Request one GPU
#flux: -g 1
# Time
#flux: --time=3h
#flux: --exclusive

# Program settings
# Where to install LLVM
export LLVMDIR=/dev/shm/rydahl1/LLVMgfx906libc
# Where to find LLVM sources
export LLVMSRCDIR=/p/lustre1/rydahl/
# Where to find the source files
export PROJECTDIR=`pwd`
# What architecture to install except for "host"
export VENDOR=AMDGPU # or NVPTX
# What subarchitecture to use
export GPUARCH=gfx906

if [[ "$VENDOR" == "AMDGPU" ]]; then
    export OMPTARGET=amdgcn-amd-amdhsa
fi
if [[ "$VENDOR" == "NVPTX" ]]; then
    export OMPTARGET=nvptx   
fi

export LD_LIBRARY_PATH=""
module load rocm
module load ninja
module load gcc/12.1.1
export CC=`which gcc`
export CXX=`which g++`

# Making bin
mkdir -p $PROJECTDIR/bin

## Cloning LLVM source files
if [ -d "$LLVMDIR/llvm-project" ] 
then
    echo "Found existing LLVM source code in $LLVMDIR"
elif [ -d "$LLVMSRCDIR/llvm-project" ] 
then
    echo "Found existing LLVM source code in $LLVMSRCDIR"
    mkdir -p $LLVMDIR/llvm-project
    cp -r $LLVMSRCDIR/llvm-project $LLVMDIR
else
    rm -rf $LLVMDIR
    mkdir -p $LLVMDIR
    cd $LLVMDIR
    git init
    git clone https://github.com/llvm/llvm-project
fi
cd $LLVMDIR/llvm-project
git checkout 3d8010363895bd063a2d33172a07985b4c9b97ee
git apply $PROJECTDIR/patches/D156263.diff

# Built-in math functions
cd $PROJECTDIR
export LIBC_GPU_VENDOR_MATH=OFF
export LIBC_GPU_BUILTIN_MATH=OFF
source install_libc.sh
cd $PROJECTDIR
./gpu_libc.sh
