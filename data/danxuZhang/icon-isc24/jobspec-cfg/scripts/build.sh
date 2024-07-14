#!/bin/bash

# ./levante-build.sh <build-dir> <options> <clean>

if [ "$3" = "true" ] && [ -d "$1" ]; then
  rm -r $1
fi

#configure muphys-cpp
cmake $2 -B $1 -S . 

#build 
cmake --build $1 --parallel