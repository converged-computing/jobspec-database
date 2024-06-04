#!/bin/bash
#PBS -q debug
#PBS -l select=1:ncpus=8

export KVS_DIR=$HOME/local/kvs
export PATH=$KVS_DIR/bin:$PATH
export KVS_OSMESA_DIR=$HOME/local/osmesa
export KVS_OSMESA_LINK_LIBRARY="-lOSMesa32 -lz `$HOME/local/llvm/bin/llvm-config --libs` `$HOME/local/llvm/bin/llvm-config --ldflags` -lrt -ldl -lpthread -lm"

source /share/G50012/local/spack/share/spack/setup-env.sh
spack load openmpi
source /share/G50012/local/OpenFOAM/OpenFOAM-2.3.1/etc/bashrc

PROGRAM=rhoPimpleFoam

#if [ -z $WM_DIR ]; then
#    . ../etc/bashrc
#fi

unset FOAM_SIGFPE
mpirun -n 8 ../${PROGRAM}_InSituVis/$PROGRAM -parallel

#if type "ffmpeg" > /dev/null 2>&1; then
#    ffmpeg -r 60 -start_number 000001 -i Output/output_%06d.bmp -vcodec libx264 -pix_fmt yuv420p -r 60 output.mp4
#fi
