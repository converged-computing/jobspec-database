#!/usr/bin/env bash

#SBATCH --job-name=arrayfire@3.7.3
#SBATCH --account=sys200
#SBATCH --partition=hotel-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -w gpu1
#SBATCH --time=01:00:00
#SBATCH --output=%x.o%j.%N
#SBATCH -x exp-15-58

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"
declare -xr TMPDIR="${LOCAL_SCRATCH_DIR}/spack-stage"
declare -xt TMP="${TMPDIR}"

declare -xr SYSTEM_NAME='tscc'

declare -xr SPACK_VERSION='0.17.3'
declare -xr SPACK_INSTANCE_NAME='gpu'
declare -xr SPACK_INSTANCE_DIR="/cm/shared/apps/spack/${SPACK_VERSION}/${SPACK_INSTANCE_NAME}"

declare -xr SLURM_JOB_SCRIPT="$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')"
declare -xr SLURM_JOB_MD5SUM="$(md5sum ${SLURM_JOB_SCRIPT})"

declare -xr SCHEDULER_MODULE='slurm'

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
module list
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"

# >> 11852    /tmp/mkandes/spack-stage/spack-stage-arrayfire-3.7.3-jgeliwmsyeyy
#              szcslsq6dv6wiyasn3da/spack-src/src/api/c/transform_coordinates.cp
#              p:108:1: fatal error: error writing to /tmp/ccHpZ2dx.s: No space 
#              left on device
#     11853      108 | }

declare -xr SPACK_PACKAGE='arrayfire@3.7.3'
declare -xr SPACK_COMPILER='gcc@8.5.0'
declare -xr SPACK_VARIANTS='+cuda cuda_arch=60,75,80,86 ~forge ~ipo ~opencl'
declare -xr SPACK_DEPENDENCIES="^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % gcc@8.5.0) ^openblas@0.3.17/$(spack find --format '{hash:7}' openblas@0.3.17 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi)"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers  
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

spack spec --long --namespaces --types arrayfire@3.7.3 % gcc@8.5.0 +cuda cuda_arch=60,75,80,86 ~forge ~ipo ~opencl "^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % gcc@8.5.0) ^openblas@0.3.17/$(spack find --format '{hash:7}' openblas@0.3.17 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi)"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all arrayfire@3.7.3 % gcc@8.5.0 +cuda cuda_arch=60,75,80,86 ~forge ~ipo ~opencl "^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % gcc@8.5.0) ^openblas@0.3.17/$(spack find --format '{hash:7}' openblas@0.3.17 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi)" 
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

spack module lmod refresh --delete-tree -y

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'libxc@5.1.5.sh'

sleep 20
