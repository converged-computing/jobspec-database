#!/bin/bash
#SBATCH --job-name=LAMMPS-ALLEGRO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=mpi
#SBATCH -o slurm-report.out
#SBATCH -e slurm-report.err

module purge
module load intel/oneapi-2023.1.0
module load icc/2023.1.0
module load mkl/2023.1.0 
module load openmpi/4.1.5

export LAMMPS_BIN=/path/to/lammps/build/bin
export PATH=$LAMMPS_BIN:$PATH

export TORCH_PATH=/home1/bastonero/builds/libtorch/1.11.0/cpu/lib/
export LD_PRELOAD="$TORCH_PATH/libtorch.so \
        $TORCH_PATH/libtorch_cpu.so \
        $TORCH_PATH/libc10.so \
        $LD_PRELOAD \
"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For parallelization flags, see: https://docs.lammps.org/Speed_kokkos.html#running-on-a-multicore-cpu

# 1 node, 64 MPI tasks/node, no multi-threading
# mpirun -np 64 lmp -k on -sf kk -in in.lj

# 1 node,  2 MPI tasks/node, 8 threads/task
# mpirun -np 32 lmp -k on t 2 -sf kk -in in.lj