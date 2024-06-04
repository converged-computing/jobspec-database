#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a01611430@unet.univie.ac.at
#SBATCH --time=10:10:00
#SBATCH --nodes=4 #equal to -N x
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --job-name AG1024
#SBATCH -o agora_1024.out
 
module load gcc/9.1.0-gcc-4.8.5-mj7s6dg
module load openmpi/3.1.4-gcc-9.1.0-fdssbx5
module load gsl/2.5-gcc-9.1.0-ucmpak4
module load fftw/3.3.8-gcc-9.1.0-2kyouz7
module load libtool/2.4.6-gcc-9.1.0-vkpnfol
module load hdf5/1.10.5-gcc-9.1.0-rolgskh
module load metis/5.1.0-gcc-9.1.0-gvmpssi
module load python/3.9.4-gcc-9.1.0-l7amfu6
module load intel-tbb/2019.4-gcc-9.1.0-sjg5kuu
module load numactl/2.0.12-gcc-9.1.0-vqkmtbi
    
cd $DATA/swiftsim/examples/agora    
mpirun -np 8 /gpfs/data/fs71636/fglatter/swiftsim/examples/swift_mpi --cosmology --self-gravity --fof --limiter --threads=24 --pin /gpfs/data/fs71636/fglatter/swiftsim/examples/agora/1024/agora_1024.yml
    
