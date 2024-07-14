#!/bin/bash

export CC=gcc
export CXX=g++

mkdir -p build
cd build
if [[ -n ${!MSYSTEM} ]]; then
	cmake -G"MSYS Makefiles" ..
else
	cmake ..
fi
make -j4

#mv "3_4" "../jobs/3_4/mpimatrix"
#mv "4_2" "../jobs/4_2/mpimatrix"
#mv "4_3" "../jobs/4_3/mpimatrix"
#mv "4_4" "../jobs/4_4/mpimatrix"
#mv "final" "../jobs/final/mpimatrix"
#mv "traffic" "../jobs/traffic/mpimatrix"