#!/usr/bin/env bash

#SBATCH --job-name=openmpi@4.1.3
#SBATCH --account=sdsc
#SBATCH --partition=hotel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%j.%N

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

declare -xr SYSTEM_NAME='expanse'

declare -xr SPACK_VERSION='0.17.3'
declare -xr SPACK_INSTANCE_NAME='cpu'
declare -xr SPACK_INSTANCE_DIR="$(HOME)/cm/shared/apps/spack/${SPACK_VERSION}/${SPACK_INSTANCE_NAME}"

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

#Error: openmpi@4.1.3%intel@19.1.1.217~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+sta
#tic+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm ^lustre@2.12.8 ^rdma-core@28.0 ^slurm@21.08.8 ^ucx@1.10.1 is unsatisfiable, conflicts are:
#  Unsatisfied conditional variants cannot be set
#  condition_requirement(2926,"version_satisfies","openmpi","1.5.4:2")
#  no version satisfies the given constraints
#  root("openmpi")
#  variant_set("openmpi","thread_multiple","True")
#  version_satisfies("openmpi","4.1.3")

#==> Error: openmpi@4.1.3%intel@19.1.1.217~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite
#3+static+vt+wrapper-rpath fabrics=ucx schedulers=slurm ^lustre@2.12.8 ^rdma-core@28.0 ^slurm@21.08.8 ^ucx@1.10.1 is unsatisfiable, conflicts are:
#  Unsatisfied conditional variants cannot be set
#  condition_requirement(2924,"version_satisfies","openmpi","1.7.3:1")
#  no version satisfies the given constraints
#  root("openmpi")
#  variant_set("openmpi","sqlite3","False")
#  version_satisfies("openmpi","4.1.3")

declare -xr INTEL_LICENSE_FILE='40000@elprado.sdsc.edu:40200@elprado.sdsc.edu'
declare -xr SPACK_PACKAGE='openmpi@4.1.3'
declare -xr SPACK_COMPILER='intel@19.1.1.217'
declare -xr SPACK_VARIANTS='~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix+romio~rsh~singularity+static+vt+wrapper-rpath cuda_arch=none fabrics=ucx schedulers=slurm'
declare -xr SPACK_DEPENDENCIES="^lustre@2.12.8 ^slurm@21.08.8 ^rdma-core ^ucx@1.10.1/$(spack find --format '{hash:7}' ucx@1.10.1 % gcc@8.5.0)"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers  
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

echo "${SPACK_SPEC}"
spack spec --long --namespaces --types openmpi@4.1.3 % intel@19.1.1.217 ~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix+romio~rsh~singularity+static+vt+wrapper-rpath cuda_arch='none' fabrics='ucx' schedulers='slurm' ^lustre@2.12.8 ^slurm@21.08.8 ^rdma-core "^ucx@1.10.1/$(spack find --format '{hash:7}' ucx@1.10.1 % gcc@8.5.0)"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all openmpi@4.1.3 % intel@19.1.1.217 ~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix+romio~rsh~singularity+static+vt+wrapper-rpath cuda_arch='none' fabrics='ucx' schedulers='slurm' ^lustre@2.12.8 ^slurm@21.08.8 ^rdma-core "^ucx@1.10.1/$(spack find --format '{hash:7}' ucx@1.10.1 % gcc@8.5.0)"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

#spack module lmod refresh --delete-tree -y
