#!/bin/bash

#SBATCH --account=ka1273
#SBATCH --job-name=scc
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.log
#SBATCH --exclusive
#SBATCH --time=00:10:00

ulimit -s unlimited
ulimit -c 0

# compiler = gnu/intel/nvidia
COMPILER='gnu' 

. scripts/levante-setup.sh $COMPILER cpu

module load cdo

# build the code
. scripts/build-cpu.sh

# run the executable
./build_single/bin/graupel $(pwd)/tasks/dbg.nc
./scripts/diffn-cpu.sh $(pwd)/reference_results/dbg_single.nc output.nc
./build_single/bin/graupel $(pwd)/tasks/input.nc
./scripts/diffn-cpu.sh $(pwd)/reference_results/sequential_single_output.nc output.nc
./build_single/bin/graupel $(pwd)/tasks/20k.nc
./scripts/diffn-cpu.sh $(pwd)/reference_results/sequential_single_20k.nc output.nc

./build_single/bin/graupel $(pwd)/tasks/1500k.nc

./build_double/bin/graupel $(pwd)/tasks/dbg.nc
./scripts/diffn-cpu.sh $(pwd)/reference_results/dbg_double.nc output.nc
./build_double/bin/graupel $(pwd)/tasks/input.nc
./scripts/diffn-cpu.sh $(pwd)/reference_results/sequential_double_output.nc output.nc
./build_double/bin/graupel $(pwd)/tasks/20k.nc
./scripts/diffn-cpu.sh $(pwd)/reference_results/sequential_double_20k.nc output.nc

./build_double/bin/graupel $(pwd)/tasks/1500k.nc
