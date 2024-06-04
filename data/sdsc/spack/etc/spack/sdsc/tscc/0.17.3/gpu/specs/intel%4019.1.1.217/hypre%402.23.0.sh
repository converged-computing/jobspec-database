#!/usr/bin/env bash

#SBATCH --job-name=hypre@2.23.0
#SBATCH --account=sys200
##SBATCH --reservation=root_73
#SBATCH --partition=hotel-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%j.%N

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

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
declare -xr COMPILER_MODULE='intel/19.1.1.217'

module purge
module load "${SCHEDULER_MODULE}"
module load ${SPACK_INSTANCE_NAME}
module load ${COMPILER_MODULE}
module list
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"

declare -xr INTEL_LICENSE_FILE='40000@elprado.sdsc.edu:40200@elprado.sdsc.edu'
declare -xr SPACK_PACKAGE='hypre@2.23.0'
declare -xr SPACK_COMPILER='intel@19.1.1.217'
declare -xr SPACK_VARIANTS='~complex +cuda cuda_arch=60,75,80,86 ~debug +fortran ~int64 ~internal-superlu ~mixedint ~mpi ~openmp +shared ~superlu-dist ~unified-memory'
declare -xr SPACK_DEPENDENCIES="^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % intel@19.1.1.217) ^intel-mkl@2020.4.304/$(spack find --format '{hash:7}' intel-mkl@2020.4.304 % ${SPACK_COMPILER} ~ilp64 threads=none)"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"
echo ${SPACK_SPEC} > spec.$$

printenv

spack config get compilers
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

spack spec --long --namespaces --types `cat spec.$$`
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all `cat spec.$$`

rm spec.$$

if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

spack module lmod refresh --delete-tree -y

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'suite-sparse@5.10.1.sh'

sleep 20
