#!/usr/bin/env bash

#SBATCH --job-name=osu-micro-benchmarks@5.7.1
#SBATCH --account=use300
##SBATCH --reservation=root_63
#SBATCH --partition=ind-gpu-shared
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
declare -xr CUDA_MODULE='cuda/11.3.1'
declare -xr MPI_MODULE='openmpi/4.1.3'

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"
module use "${SPACK_ROOT}/share/spack/lmod/linux-rocky8-x86_64/Core"
module load "${COMPILER_MODULE}"
module load "${CUDA_MODULE}"
module load "${MPI_MODULE}"
module list

# omb package cannot find libcuda since it is located with cuda system/driver installation, not the spack cuda install; package.py file needs to be updated to allow splitting of the two sets of libraries.
# ==> No patches needed for osu-micro-benchmarks
#==> osu-micro-benchmarks: Executing phase: 'autoreconf'
#==> osu-micro-benchmarks: Executing phase: 'configure'
#==> Error: ProcessError: Command exited with status 1:
#    '/tmp/mkandes/spack-stage/spack-stage-osu-micro-benchmarks-5.7.1-qrveszohcnipx7bn6kcflfty2d5hf6we/spack-src/configure' '--prefix=/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/osu-micro-benchmarks-5.7.1-qrveszohcnipx7bn6kcflfty2d5hf6we' 'CC=/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/openmpi-4.1.3-5lmvx2qm7zynusyafocgqddpjnv3qegm/bin/mpicc' 'CXX=/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/openmpi-4.1.3-5lmvx2qm7zynusyafocgqddpjnv3qegm/bin/mpic++' '--enable-cuda' '--with-cuda=/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/cuda-11.3.1-gonjgx5gtwrgpnixvmchcaozt6bv2ykl' 'LDFLAGS=-lrt'
#
#1 error found in build log:
#     106    checking for MPI_Get_accumulate... yes
#     107    checking for shmem_barrier_all... no
#     108    checking for upc_memput... no
#     109    checking whether upcxx_alltoall is declared... no
#     110    checking for shmem_finalize... no
#     111    checking for library containing cuPointerGetAttribute... no
#  >> 112    configure: error: cannot link with -lcuda

declare -xr SPACK_PACKAGE='osu-micro-benchmarks@5.7.1'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='~cuda'
declare -xr SPACK_DEPENDENCIES="^openmpi@4.1.3/$(spack find --format '{hash:7}' openmpi@4.1.3 % ${SPACK_COMPILER})"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers  
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

spack spec --long --namespaces --types "${SPACK_SPEC}"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all "${SPACK_SPEC}"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

spack module lmod refresh --delete-tree -y

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'netlib-scalapack@2.1.0.sh'

sleep 60
