#!/bin/sh 
#PBS -l walltime=2:00:00
#PBS -lselect=1:ncpus=8:mem=1024MB
#PBS -j oe
#PBS -q tng


# get modules and directories
module load lammps/6Dec12 mpi intel-suite
#module load lammps/14May13 mpi intel-suite fftw/2.1.5-double 

WRK="$PBS_O_WORKDIR"
TMP="$TMPDIR"


# push
cp -v ${WRK}/*     ${TMP}


# run
pushd ${TMP}

n=0; mpiexec lammps -sc none < ${n}.inp.run && \
n=1; mpiexec lammps -sc none < ${n}.inp.run

popd

# pull
cp -v ${TMP}/*     ${WRK}
