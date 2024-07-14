#!/bin/bash -l
if [[ "${SLURM_CLUSTER_NAME}" == "escori" ]]; then
    module purge
    module load dgx
    module load nvhpc/21.7
    module load cuda/11.2.1
    module list
    cpus=${SLURM_CPUS_PER_TASK:-32}
    RUN="srun -n 1 -c ${cpus} --cpu-bind=cores"
fi
set -x
set -e

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

FCFLAGS="-fast -Mpreprocess -Wall -Wextra -mp=gpu -gpu=cc80 -Minfo=mp,accel"

# NOTE - The clone is not needed since the repo is added as a submodule.
#if [ ! -d berkeleygw-kernels-fortran ]; then
#    git clone --single-branch --branch master git@gitlab.com:NERSC/nersc-proxies/berkeleygw-kernels-fortran.git
#fi
cd berkeleygw-kernels-fortran
git checkout a31a84de99c86becd3fab6385c2a3d4f8ddd7417

# make ARCH=A100
for CPPFLAGS in "-UUSE_LOOP" "-DUSE_LOOP" "-DUSE_LOOP -DTHREAD_LIMIT=1024"; do
    echo "Using CPPFLAGS=${CPPFLAGS}"
    nvfortran ${CPPFLAGS} ${FCFLAGS} -c gpp_data8.f90 -o gpp_data8.o
    nvfortran ${CPPFLAGS} ${FCFLAGS} -c gpp8.f90 -o gpp8.o
    nvfortran ${FCFLAGS} gpp8.o gpp_data8.o -o gpp8.x
    for i in {1..10}; do ${RUN} ./gpp8.x; done
    rm -f gpp8.o gpp8.x gpp_data8.o sigma_gpp_data_module.mod
done
