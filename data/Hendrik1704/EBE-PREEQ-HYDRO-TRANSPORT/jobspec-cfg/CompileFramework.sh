#!/bin/bash

# Determine the number of available CPUs
NUM_CPUS=$(nproc)

echo "Build KoMPoST code for pre-equilibrium stage"
cd KoMPoST && make -j$NUM_CPUS && cd ..

echo "Build MUSIC code for hydrodynamic stage"
cd MUSIC && mkdir build && cd build && cmake .. && make -Bj$NUM_CPUS && make install && cd .. && rm -r build && cd ..

echo "Build iSS sampler to perform freeze out"
cd iSS && mkdir -p build && cd build && rm -fr * && cmake .. && make -Bj$NUM_CPUS && make install && cd .. && rm -fr build && cd ..

echo "Build Pythia which is used in SMASH for the hadronic afterburner phase"
tar xf pythia8310.tgz
cd pythia8310
./configure --cxx-common='-std=c++17 -march=native -O3 -fPIC -pthread'
make -j$NUM_CPUS && cd ..

echo "Prepare the Eigen library for SMASH"
tar -xf eigen-3.4.0.tar.gz

echo "Build SMASH as an hadronic afterburner"
cd smash && mkdir build && cd build && cmake .. -DTRY_USE_ROOT=OFF -DTRY_USE_HEPMC=OFF -DPythia_CONFIG_EXECUTABLE=../../pythia8310/bin/pythia8-config -DCMAKE_PREFIX_PATH=../../eigen-3.4.0/ && make smash -j$NUM_CPUS

echo "Finished building all the modules successfully"

