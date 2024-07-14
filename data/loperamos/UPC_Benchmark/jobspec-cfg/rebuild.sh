#!/bin/bash
mkdir build
mkdir build/bin
cd build
#rm -fr *

if [ -z "$2" ] ; then
threads=16
else
threads=$2
fi

if [ $1 == "R" ] ; then
    cmake .. -DTEST_TO_RUN=affinity_vs_T -DTHREADS=$threads -DCMAKE_VERBOSE_MAKEFILE=OFF -DCMAKE_BUILD_TYPE=Release
    make
else    
    echo Debugging
    cmake .. -DTEST_TO_RUN=affinity_vs_T -DTHREADS=$threads -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Debug
    make verbose=1
fi

cd ../