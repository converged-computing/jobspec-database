#!/usr/bin/env bash

#SBATCH --job-name=openmm@7.5.0
#SBATCH --account=sdsc
#SBATCH --partition=hotel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=93G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%j.%N

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

declare -xr SYSTEM_NAME='expanse'

declare -xr SPACK_VERSION='0.17.3'
declare -xr SPACK_INSTANCE_NAME='gpu'
declare -xr SPACK_INSTANCE_DIR="${HOME}/cm/shared/apps/spack/${SPACK_VERSION}/${SPACK_INSTANCE_NAME}"

declare -xr SLURM_JOB_SCRIPT="$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')"
declare -xr SLURM_JOB_MD5SUM="$(md5sum ${SLURM_JOB_SCRIPT})"

declare -xr SCHEDULER_MODULE='slurm'
declare -xr COMPILER_MODULE='gcc/10.2.0'
declare -xr CUDA_MODULE='cuda/11.2.2'

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"
module use "${SPACK_ROOT}/share/spack/lmod/linux-rocky8-x86_64/Core"
module load "${COMPILER_MODULE}"
module load "${CUDA_MODULE}"
module list

# 1 error found in build log:
#     43      cmake_modules/FindOpenCL.cmake:86 (find_package_handle_standard_ar
#           gs)
#     44      CMakeLists.txt:338 (FIND_PACKAGE)
#     45    This warning is for project developers.  Use -Wno-dev to suppress it
#           .
#     46    
#     47    -- Found OPENCL: /home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/s
#           pack/linux-rocky8-cascadelake/gcc-10.2.0/cuda-11.2.2-blza2psofa3wr2z
#           umqrnh4je2f7ze3mx/lib64/libOpenCL.so
#     48    -- Configuring done
#  >> 49    CMake Error: The following variables are used in this project, but t
#           hey are set to NOTFOUND.
#     50    Please set them or make sure they are set and tested correctly in th
#           e CMake files:
#     51    CUDA_CUDA_LIBRARY (ADVANCED)
#     52        linked by target "OpenMMCUDA" in directory /tmp/mkandes/spack-st
#           age/spack-stage-openmm-7.5.0-fjiqzymyngtbtisos5koguvos4c3urjv/spack-
#           src/platforms/cuda/sharedTarget
#     53    
#     54    -- Generating done
#     55    CMake Generate step failed.  Build files cannot be regenerated corre
#           ctly.
#
# FIX: https://github.com/floydhub/dl-docker/pull/48
declare -xr CUDA_CUDA_LIBRARY='/cm/local/apps/cuda/libs/current/lib64'
declare -xr CMAKE_LIBRARY_PATH="${CUDA_CUDA_LIBRARY}"

# /tmp/mkandes/spack-stage/spack-stage-openmm-7.5.0-fjiqzymyngtbtisos5koguvos4c3urjv/spack-build-fjiqzym/wrappers/OpenMMCWrapper.h:389:41: error: conflicting declaration of C function 'OpenMM_StringArray OpenMM_Platform_getPluginLoadFailures()'
#  389 | extern OPENMM_EXPORT OpenMM_StringArray OpenMM_Platform_getPluginLoadFailures();
#      |                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# /tmp/mkandes/spack-stage/spack-stage-openmm-7.5.0-fjiqzymyngtbtisos5koguvos4c3urjv/spack-build-fjiqzym/wrappers/OpenMMCWrapper.h:163:42: note: previous declaration 'OpenMM_StringArray* OpenMM_Platform_getPluginLoadFailures()'
#  163 | extern OPENMM_EXPORT OpenMM_StringArray* OpenMM_Platform_getPluginLoadFailures();
#      |                                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Fix? https://github.com/openmm/openmm/issues/3656
declare -xr OPENMM_PLUGIN_DIR='/tmp/mkandes/spack-stage/spack-stage-openmm-7.5.0-fjiqzymyngtbtisos5koguvos4c3urjv/spack-src/plugins'

declare -xr SPACK_PACKAGE='openmm@7.5.0'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='+cuda cuda_arch=70,80 ~ipo'
declare -xr SPACK_DEPENDENCIES="^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % ${SPACK_COMPILER}) ^py-numpy@1.20.3/$(spack find --format '{hash:7}' py-numpy@1.20.3 % ${SPACK_COMPILER}) ^openblas@0.3.17/$(spack find --format '{hash:7}' openblas@0.3.17 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp)"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers  
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

spack spec --long --namespaces --types openmm@7.5.0 % gcc@10.2.0 +cuda cuda_arch=70,80 ~ipo "${SPACK_DEPENDENCIES}"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all openmm@7.5.0 % gcc@10.2.0 +cuda cuda_arch=70,80 ~ipo "${SPACK_DEPENDENCIES}"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

spack module lmod refresh --delete-tree -y

#sbatch --dependency="afterok:${SLURM_JOB_ID}" ''

sleep 60
