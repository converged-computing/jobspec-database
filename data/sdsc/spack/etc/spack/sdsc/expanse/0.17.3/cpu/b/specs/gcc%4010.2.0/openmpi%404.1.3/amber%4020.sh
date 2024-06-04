#!/usr/bin/env bash

#SBATCH --job-name=amber@20
#SBATCH --account=use300
#SBATCH --reservation=rocky8u7_testing
#SBATCH --partition=ind-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x.o%j.%N

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"
declare -xr TMPDIR="${LOCAL_SCRATCH_DIR}"

declare -xr SYSTEM_NAME='expanse'

declare -xr SPACK_VERSION='0.17.3'
declare -xr SPACK_INSTANCE_NAME='cpu'
declare -xr SPACK_INSTANCE_VERSION='b'
declare -xr SPACK_INSTANCE_DIR="/cm/shared/apps/spack/${SPACK_VERSION}/${SPACK_INSTANCE_NAME}/${SPACK_INSTANCE_VERSION}"

declare -xr SLURM_JOB_SCRIPT="$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')"
declare -xr SLURM_JOB_MD5SUM="$(md5sum ${SLURM_JOB_SCRIPT})"

declare -xr SCHEDULER_MODULE='slurm'
declare -xr COMPILER_MODULE='gcc/10.2.0'
declare -xr MPI_MODULE='openmpi/4.1.3'
#declare -xr CUDA_MODULE='cuda/11.2.2'
declare -xr CMAKE_MODULE='cmake/3.21.4'

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"
module use "${SPACK_ROOT}/share/spack/lmod/linux-rocky8-x86_64/Core"
module load "${COMPILER_MODULE}"
module load "${MPI_MODULE}"
#module load "${CUDA_MODULE}"
module load "${CMAKE_MODULE}"
module list

# 3 errors found in build log:
#     1514    [ 17%] Building C object AmberTools/src/fftw-3.3/rdft/scalar/r2cf/
#             CMakeFiles/rdft_scalar_r2cf.dir/r2cfII_20.c.o
#     1515    [ 17%] Building C object AmberTools/src/fftw-3.3/api/CMakeFiles/ff
#             tw_api.dir/plan-many-r2r.c.o
#     1516    [ 17%] Building C object AmberTools/src/fftw-3.3/dft/simd/CMakeFil
#             es/dft_sse2_codelets.dir/sse2/n2sv_64.c.o
#     1517    CMakeFiles/cifparse.dir/lex.cif.c.o:(.bss+0x8): multiple definitio
#             n of `cifpin'
#     1518    CMakeFiles/cifparse.dir/cifparse.c.o:(.bss+0x0): first defined her
#             e
#     1519    Manifying 2 pod documents
#  >> 1520    collect2: error: ld returned 1 exit status
#  >> 1521    make[2]: *** [AmberTools/src/cifparse/CMakeFiles/cifparse.dir/buil
#             d.make:141: AmberTools/src/cifparse/libcifparse.so] Error 1
#  >> 1522    make[1]: *** [CMakeFiles/Makefile2:4019: AmberTools/src/cifparse/C
#             MakeFiles/cifparse.dir/all] Error 2
#     1523    make[1]: *** Waiting for unfinished jobs....
#     1524    /scratch/spack_cpu/job_21100117/spack-stage/spack-stage/spack-stag
#             e-amber-20-qdzf4ls6iw56fqda3hncmqyldoqiubs6/spack-src/AmberTools/s
#             rc/blas/dtrmv.f:190:0:
#     1525
#     1526      190 |                   JX = JX + INCX
#     1527          |
#     1528    Warning: 'kx' may be used uninitialized in this function [-Wmaybe-
#             uninitialized]

declare -xr SPACK_PACKAGE='amber@20'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='~cuda +mpi +openmp +update'
declare -xr SPACK_DEPENDENCIES="^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi) ^openmpi@4.1.3/$(spack find --format '{hash:7}' openmpi@4.1.3 % ${SPACK_COMPILER}) ^netcdf-c@4.8.1/$(spack find --format '{hash:7}' netcdf-c@4.8.1 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3)"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

time -p spack spec --long --namespaces --types "${SPACK_SPEC}"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all "${SPACK_SPEC}"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

#spack module lmod refresh --delete-tree -y

#sbatch --dependency="afterok:${SLURM_JOB_ID}" ''

sleep 30
