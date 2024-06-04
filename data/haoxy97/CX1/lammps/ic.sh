#! /bin/bash

# Install a modified version lammps in CX1
# must compile at the login node. need esential libs.
module purge
module load tools/eb-dev
module load libreadline
module load aocc/2.3.0

cd ${HOME}
mkdir -p ${HOME}/openmpi4
mkdir -p ${HOME}/lammps

cd ${HOME}
wget 'https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz'
tar -xvf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5
./configure CC=clang CCX=clang++ FC=flang --prefix=${HOME}/openmpi4 --enable-orterun-prefix-by-default --enable-mpi-cxx
make -j 2
make install
export PATH="${HOME}/openmpi4/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/openmpi4/lib:$LD_LIBRARY_PATH"


cd ${HOME}
wget 'https://download.lammps.org/tars/lammps-stable.tar.gz'
mkdir -p lammps-stable
tar -xvf lammps-stable.tar.gz -C lammps-stable --strip-components=1
cd lammps-stable
mkdir build
cd build

cmake -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH=${HOME}/openmpi4/lib  -D PKG_OPENMP=yes -D PKG_OPT=yes -D BUILD_LAMMPS_SHELL=yes  -D LAMMPS_EXCEPTIONS=yes  -D BUILD_MPI=yes\
 -D PKG_MOLECULE=yes  -D PKG_KSPACE=yes -D PKG_EXTRA-PAIR=YES  -D PKG_RIGID=YES -D CMAKE_INSTALL_PREFIX=${HOME}/lammps ../cmake
make -j 2
make install

export PATH="${HOME}/lammps/bin:$PATH"

echo try input lmp to test it.
echo example of your pbs.sh file
echo '''------------------------------------
#!/bin/bash
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=71:00:00

cd $PBS_O_WORKDIR

module purge
module load tools/eb-dev
module load libreadline
module load aocc/2.3.0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="${HOME}/lammps/bin:${HOME}/openmpi4/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/openmpi4/lib:$LD_LIBRARY_PATH"

mpirun -np 4 lmp -i lmp.in
------------------------------------'''

