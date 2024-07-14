#!/bin/bash
set -o nounset

cd $LLVMDIR
rm -rf $LLVMDIR/build; mkdir $LLVMDIR/build
rm -rf $LLVMDIR/install; mkdir $LLVMDIR/install
cd $LLVMDIR/build

cmake \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$LLVMDIR/install \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$LD_LIBRARY_PATH" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_LIT_ARGS=-v \
        -DLLVM_LIBC_FULL_BUILD=1 \
        -DLLVM_TARGETS_TO_BUILD="host;$VENDOR" \
        -DLLVM_ENABLE_PROJECTS="clang;lld;openmp" \
        -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
	-DLIBOMPTARGET_ENABLE_DEBUG=ON  \
        -DLIBC_GPU_ARCHITECTURES=$GPUARCH \
	-DLIBC_GPU_TEST_ARCHITECTURE=$GPUARCH \
        -DLIBC_GPU_VENDOR_MATH=$LIBC_GPU_VENDOR_MATH \
        -DLIBC_GPU_BUILTIN_MATH=$LIBC_GPU_BUILTIN_MATH \
        ../llvm-project/llvm

ninja install -j 32

export PATH=$LLVMDIR/install/bin/:$PATH
export LD_LIBRARY_PATH=$LLVMDIR/install/lib/:$LD_LIBRARY_PATH
