#!/bin/bash
#PBS -l walltime=96:00:00             # WALLTIME limit
#PBS -q long                          # short queue
#PBS -l nodes=16:ppn=24               # Number of nodes, put 24 processes on each
#PBS -N ew_p1_df                      # Name of job
#PBS -A windsim                       # Project handle
 
cd $PBS_O_WORKDIR
module purge
module load gcc/5.2.0 python/2.7.8 &> /dev/null 
export SPACK_ROOT=/projects/windsim/exawind/SharedSoftware/spack 
export COMPILER=gcc 
export PATH=${SPACK_ROOT}/bin/:$PATH 
module use ${SPACK_ROOT}/share/spack/modules/$(${SPACK_ROOT}/bin/spack arch) 
module load $(spack module find -m tcl cmake %${COMPILER}) 
module load $(spack module find -m tcl openmpi %${COMPILER}) 
module load $(spack module find -m tcl hdf5 %${COMPILER}) 
module load $(spack module find -m tcl zlib %${COMPILER}) 
module load $(spack module find -m tcl libxml2 %${COMPILER}) 
module load $(spack module find -m tcl binutils %${COMPILER}) 
module list
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gvijayak/CurrentProjects/OpenFAST/install/lib

/home/gvijayak/CurrentProjects/NaluWindUtils/build/src/mesh/abl_mesh -i gen_mesh.i &> log.mesh
mpirun -np 372 /home/gvijayak/CurrentProjects/Nalu/build.actLineHO/naluX -i ellipticWing.i --pprint &> log.run
