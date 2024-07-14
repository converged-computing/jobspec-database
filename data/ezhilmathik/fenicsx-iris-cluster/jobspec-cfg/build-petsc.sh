#!/bin/bash -l
set -e
source env-build-fenics.sh

VERSION="3.16.1"

mkdir -p $BUILD_DIR

cd ${BUILD_DIR} && \
   wget --no-check-certificate --read-timeout=10 -nc http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${VERSION}.tar.gz -O petsc.tar.gz && \
   mkdir -p ${BUILD_DIR}/petsc && \
   tar -xf petsc.tar.gz -C ${BUILD_DIR}/petsc --strip-components=1 && \
   cd petsc && \
   python3 ./configure \
     --with-blaslapack-dir=${EBROOTOPENBLAS}/lib \
     --COPTFLAGS="${FLAGS}" \
     --CXXOPTFLAGS="${FLAGS}" \
     --FOPTFLAGS="${FLAGS}" \
     --with-cc=${MPICC} --with-cxx=${MPICXX} --with-fc=${MPIFC} \
     --with-make-np=16 \
     --with-mpiexec="srun" \
     --download-metis \
     --download-ptscotch \
     --download-parmetis \
     --download-suitesparse \
     --download-scalapack \
     --download-hypre \
     --download-mumps \
     --download-ml \
     --download-superlu \
     --download-superlu_dist \
     --with-scalar-type=real \
     --with-debugging=0 \
     --with-shared-libraries \
     --with-fortran-bindings=no \
     --prefix=${PREFIX} && \
    make && \
    make install
