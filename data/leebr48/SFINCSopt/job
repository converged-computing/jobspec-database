#!/bin/bash -l

# Job Name:
#SBATCH -J sfincsOpt

# Standard output and error:
#SBATCH -o ./sfincsOpt.out.%j
#SBATCH -e ./sfincsOpt.err.%j
# Initial working directory:
#SBATCH -D ./

#SBATCH --nodes=18
#SBATCH --ntasks=720
#SBATCH --cpus-per-task=1

#SBATCH --mail-type=all
#SBATCH --mail-user=brandon.lee@ipp.mpg.de

# Wall clock limit:
#SBATCH --time=24:00:00

module purge
module load intel/19.1.2
module load mkl/2020.4
module load impi/2019.8
module load hdf5-mpi/1.8.22
module load netcdf-mpi/4.4.1
module load fftw-mpi
module load anaconda/3/2020.02
module load petsc-real/3.13.5
module load mumps-32-noomp/5.1.2
module load gcc/11

python main.py
