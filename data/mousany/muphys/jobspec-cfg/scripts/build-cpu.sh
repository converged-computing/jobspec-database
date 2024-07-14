#!/bin/bash

rm -rf build_single build_double

cmake -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CXX_FLAGS="-O0" \
      -DMU_IMPL=omp \
      -DMU_ENABLE_SINGLE=ON \
      -B build_single -S . \
      && cmake --build build_single

cmake -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CXX_FLAGS="-O0" \
        -DMU_IMPL=omp \
        -DMU_ENABLE_SINGLE=OFF \
        -B build_double -S . \
        && cmake --build build_double
