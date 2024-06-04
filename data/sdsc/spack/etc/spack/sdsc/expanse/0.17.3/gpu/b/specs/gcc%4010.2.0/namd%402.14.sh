#!/usr/bin/env bash

#SBATCH --job-name=namd@2.14
#SBATCH --account=use300
#SBATCH --reservation=rocky8u7_testing
#SBATCH --partition=ind-gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=93G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.o%j.%N

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"
declare -xr TMPDIR="${LOCAL_SCRATCH_DIR}"

declare -xr SYSTEM_NAME='expanse'

declare -xr SPACK_VERSION='0.17.3'
declare -xr SPACK_INSTANCE_NAME='gpu'
declare -xr SPACK_INSTANCE_VERSION='b'
declare -xr SPACK_INSTANCE_DIR="/cm/shared/apps/spack/${SPACK_VERSION}/${SPACK_INSTANCE_NAME}/${SPACK_INSTANCE_VERSION}"

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


# smp is not added to CHARMARCH as should be expected, despite charmpp being built with verbs +smp; one of these packages may need to be fixed, or both?

#==> namd: Executing phase: 'edit'
#==> [2022-12-16-09:03:05.607692] '/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/gcc-10.2.0-i62tgsoexc6ya4h7urwhriiujk22nrnj/bin/gcc' '-dumpversion'
#==> [2022-12-16-09:03:05.633662] Copying arch/Linux-x86_64.base to arch/linux-x86_64.base
#==> [2022-12-16-09:03:05.634121] Copying arch/Linux-x86_64.fftw3 to arch/linux-x86_64.fftw3
#==> [2022-12-16-09:03:05.634455] Copying arch/Linux-x86_64.tcl to arch/linux-x86_64.tcl
#==> [2022-12-16-09:03:05.634776] FILTER FILE: arch/linux-x86_64.tcl [replacing "-ltcl8\.5"]
#==> [2022-12-16-09:03:05.635607] Copying arch/Linux-x86_64.cuda to arch/linux-x86_64.cuda
#==> [2022-12-16-09:03:05.635913] FILTER FILE: arch/linux-x86_64.cuda [replacing "^CUDADIR=.*$"]
#==> [2022-12-16-09:03:05.636624] './config' 'linux-x86_64-spack' '--charm-base' '/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/charmpp-6.10.2-nlla43jibnyqgezncn7q44cpzoddjlxe' '--with-fftw3' '--fftw-prefix' '/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/fftw-3.3.10-7ahyh5v5twqbte3ye62ukz55jqgwheuk' '--with-tcl' '--tcl-prefix' '/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/tcl-8.5.9-6xpusrkzpnmqhi3uw6anyn7unxlsqvty' '--with-cuda' '--cuda-prefix' '/home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-cascadelake/gcc-10.2.0/cuda-11.3.1-gonjgx5gtwrgpnixvmchcaozt6bv2ykl'
#
#Selected arch file arch/linux-x86_64-spack.arch contains:
#
#NAMD_ARCH = linux-x86_64
#CHARMARCH = verbs-linux-x86_64
#CXX = /home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/gcc-10.2.0-i62tgsoexc6ya4h7urwhriiujk22nrnj/bin/g++ -std=c++11
#CXXOPTS = -m64 -O3 -fexpensive-optimizations                                         -ffast-math -lpthread -march=cascadelake -mtune=cascadelake
#CC = /home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/gcc-10.2.0-i62tgsoexc6ya4h7urwhriiujk22nrnj/bin/gcc
#COPTS = -m64 -O3 -fexpensive-optimizations                                         -ffast-math -lpthread -march=cascadelake -mtune=cascadelake
#
#ERROR: Non-SMP Charm++ arch  is not compatible with CUDA NAMD.
#ERROR: CUDA builds require non-MPI SMP or multicore Charm++ arch for reasonable performance.
#
#Consider ucx-smp or verbs-smp (InfiniBand), gni-smp (Cray), or multicore (single node).

declare -xr SPACK_PACKAGE='namd@2.14'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='+cuda cuda_arch=70,80 interface=tcl'
declare -xr SPACK_DEPENDENCIES="^charmpp@6.10.2/$(spack find --format '{hash:7}' charmpp@6.10.2 % ${SPACK_COMPILER} backend=multicore) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^tcl@8.5.9/$(spack find --format '{hash:7}' tcl@8.5.9 % ${SPACK_COMPILER})"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

time -p spack spec --long --namespaces --types namd@2.14 % gcc@10.2.0 +cuda cuda_arch=70,80 interface=tcl "^charmpp@6.10.2/$(spack find --format '{hash:7}' charmpp@6.10.2 % ${SPACK_COMPILER} backend=multicore) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^tcl@8.5.9/$(spack find --format '{hash:7}' tcl@8.5.9 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all namd@2.14 % gcc@10.2.0 +cuda cuda_arch=70,80 interface=tcl "^charmpp@6.10.2/$(spack find --format '{hash:7}' charmpp@6.10.2 % ${SPACK_COMPILER} backend=multicore) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^tcl@8.5.9/$(spack find --format '{hash:7}' tcl@8.5.9 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

#spack module lmod refresh --delete-tree -y

#sbatch --dependency="afterok:${SLURM_JOB_ID}" ''

sleep 30
