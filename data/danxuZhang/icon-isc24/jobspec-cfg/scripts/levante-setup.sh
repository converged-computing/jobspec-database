#!/bin/bash

c1="gnu"
c2="intel"
c3="nvidia"

a1="cpu"
a2="gpu"

spack load netcdf-cxx4@4.3.1

echo "compilation for architecture [$2] using compilation chain [$1]"
echo ""

if [[ "$1" == "$c1" ]]; then
    module load gcc
elif [[ "$1" == "$c2" ]]; then
    module load intel-oneapi-compilers/2022.0.1-gcc-11.2.0
elif [[ "$1" == "$c3" ]]; then
    module load nvhpc/23.7-gcc-11.2.0
else 
    echo "$1 is not supported!"
fi

if [[ "$2" == "$a2" ]]; then 
    module load nvhpc/23.7-gcc-11.2.0
fi