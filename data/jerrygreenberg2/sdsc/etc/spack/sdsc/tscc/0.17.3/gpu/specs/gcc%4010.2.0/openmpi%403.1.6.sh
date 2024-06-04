#!/usr/bin/env bash

#SBATCH --job-name=openmpi@3.1.6
#SBATCH --account=sdsc
#SBATCH --partition=hotel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=93G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
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

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
module list
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"

# ==> Error: openmpi@3.1.6%gcc@10.2.0~atomics+cuda+cxx+cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix+romio~rsh~singularity+static+v
#t+wrapper-rpath cuda_arch=70 fabrics=verbs schedulers=slurm ^cuda@11.2.2%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=nat
#ive" ~dev arch=linux-rocky8-cascadelake ^libiconv@1.16%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native"  libs=shared,stat
#ic arch=linux-rocky8-cascadelake ^libxml2@2.9.12%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" ~python arch=linux-rock
#y8-cascadelake ^lustre@2.12.8 ^rdma-core@28.0 ^slurm@21.08.8 ^xz@5.2.5%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" ~
#pic libs=shared,static arch=linux-rocky8-cascadelake ^zlib@1.2.11%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" +optim
#ize+pic+shared arch=linux-rocky8-cascadelake is unsatisfiable, conflicts are:
#  condition(2923)
#  condition(3139)
#  dependency_type(3139,"link")
#  imposed_constraint(3139,"version_satisfies","cuda","11.0:")
#  no version satisfies the given constraints
#  root("openmpi")
#  variant_condition(2923,"openmpi","cuda_arch")
#  variant_set("openmpi","cuda_arch","80")
#  version("cuda","10.2.89")
#  version_satisfies("cuda","10.1.243:","10.2.89")
#  version_satisfies("cuda","10.1.243:","11.0.2")
#  version_satisfies("cuda","10.1.243:","11.0.3")
#  version_satisfies("cuda","10.1.243:","11.1.0")
#  version_satisfies("cuda","10.1.243:","11.1.1")
#  version_satisfies("cuda","10.1.243:","11.2.0")
#  version_satisfies("cuda","10.1.243:","11.2.1")
#  version_satisfies("cuda","10.1.243:","11.2.2")
#  version_satisfies("cuda","10.1.243:","11.3.0")
#  version_satisfies("cuda","10.1.243:","11.3.1")
#  version_satisfies("cuda","10.1.243:","11.4.0")
#  version_satisfies("cuda","10.1.243:","11.4.1")
#  version_satisfies("cuda","10.1.243:","11.4.2")
#  version_satisfies("cuda","10.1.243:","11.5.0")

# ==> Error: openmpi@3.1.6%gcc@10.2.0~atomics+cuda+cxx+cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix+romio~rsh~singularity+static+vt+wrapper-rpath cuda_arch=70 fabrics=verbs schedulers=slurm ^cuda@11.2.2%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native " ~dev arch=linux-rocky8-cascadelake ^libiconv@1.16%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native"  libs=shared,static arch=linux-rocky8-cascadelake ^libxml2@2.9.12%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" ~python arch=linux-rocky8-cascadelake ^lustre@2.12.8 ^rdma-core@28.0 ^slurm@21.08.8 ^xz@5.2.5%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" ~pic libs=shared,static arch=linux-rocky8-cascadelake ^zlib@1.2.11%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" +optimize+pic+shared arch=linux-rocky8-cascadelake is unsatisfiable, conflicts are:
#
#  A conflict was triggered
#  condition(2922)
#  condition(2970)
#  condition(2971)
#  condition(3119)
#  conflict("openmpi",2970,2971)
#  dependency_condition(3119,"openmpi","cuda")
#  dependency_type(3119,"link")
#  node_compiler_version_satisfies("openmpi","gcc","9:","10.2.0")
#  node_compiler_version_set("openmpi","gcc","10.2.0")
#  node_target_satisfies("cuda","x86_64:","broadwell")
#  node_target_satisfies("cuda","x86_64:","bulldozer")
#  node_target_satisfies("cuda","x86_64:","cannonlake")
#  node_target_satisfies("cuda","x86_64:","cascadelake")
#  node_target_satisfies("cuda","x86_64:","core2")
#  node_target_satisfies("cuda","x86_64:","excavator")
#  node_target_satisfies("cuda","x86_64:","haswell")
#  node_target_satisfies("cuda","x86_64:","icelake")
#  node_target_satisfies("cuda","x86_64:","ivybridge")
#  node_target_satisfies("cuda","x86_64:","k10")
#  node_target_satisfies("cuda","x86_64:","mic_knl")
#  node_target_satisfies("cuda","x86_64:","nehalem")
#  node_target_satisfies("cuda","x86_64:","nocona")
#  node_target_satisfies("cuda","x86_64:","piledriver")
#  node_target_satisfies("cuda","x86_64:","sandybridge")
#  node_target_satisfies("cuda","x86_64:","skylake")
#  node_target_satisfies("cuda","x86_64:","skylake_avx512")
#  node_target_satisfies("cuda","x86_64:","steamroller")
#  node_target_satisfies("cuda","x86_64:","westmere")
#  node_target_satisfies("cuda","x86_64:","x86_64")
#  node_target_satisfies("cuda","x86_64:","x86_64_v2")
#  node_target_satisfies("cuda","x86_64:","x86_64_v3")
#  node_target_satisfies("cuda","x86_64:","x86_64_v4")
#  node_target_satisfies("cuda","x86_64:","zen")
#  node_target_satisfies("cuda","x86_64:","zen2")
#  node_target_satisfies("cuda","x86_64:","zen3")
#  root("openmpi")
#  variant_condition(2922,"openmpi","cuda")
#  variant_set("openmpi","cuda","True")
#  version("cuda","10.2.89")
#  version_satisfies("cuda",":10.2.89","10.2.89")

declare -xr SPACK_PACKAGE='openmpi@3.1.6'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='~atomics +cuda cuda_arch=70,80 +cxx +cxx_exceptions ~gpfs ~internal-hwloc ~java +legacylaunchers +lustre ~memchecker +pmi +pmix +romio ~rsh ~singularity +static +vt +wrapper-rpath fabrics=verbs schedulers=slurm'
declare -xr SPACK_DEPENDENCIES="^lustre@2.12.8 ^slurm@21.08.8 ^rdma-core@28.0 ^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % ${SPACK_COMPILER})"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers  
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

spack spec --long --namespaces --types openmpi@3.1.6 % gcc@10.2.0 ~atomics +cuda cuda_arch=70,80 +cxx +cxx_exceptions ~gpfs ~internal-hwloc ~java +legacylaunchers +lustre ~memchecker +pmi +pmix +romio ~rsh ~singularity +static +vt +wrapper-rpath fabrics=verbs schedulers=slurm "^lustre@2.12.8 ^slurm@21.08.8 ^rdma-core@28.0 ^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all openmpi@3.1.6 % gcc@10.2.0 ~atomics +cuda cuda_arch=70,80 +cxx +cxx_exceptions ~gpfs ~internal-hwloc ~java +legacylaunchers +lustre ~memchecker +pmi +pmix +romio ~rsh ~singularity +static +vt +wrapper-rpath fabrics=verbs schedulers=slurm "^lustre@2.12.8 ^slurm@21.08.8 ^rdma-core@28.0 ^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

spack module lmod refresh --delete-tree -y
