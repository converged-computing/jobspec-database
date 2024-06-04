#!/usr/bin/env bash

#SBATCH --job-name=ucx@1.10.1
#SBATCH --account=use300
##SBATCH --reservation=root_63
#SBATCH --partition=ind-gpu-shared
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

# ==> Error: ucx@1.10.1%gcc@10.2.0~assertions~cm+cma+cuda+dc~debug+dm+gdrcopy+ib-hw-tm~java~knem~logging+mlx5-dv+optimizations~parameter_checking+pic+rc~rocm+thread_mul tiple+ud~xpmem cuda_arch=70 ^cuda@11.2.2%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" ~dev arch=linux-rocky8-cascadelake ^libiconv@1.16%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native"  libs=shared,static arch=linux-rocky8-cascadelake ^libxml2@2.9.12%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" ~python arch=linux-rocky8-cascadelake ^xz@5.2.5%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" ~pic libs=shared,static arch=linux-rocky8-cascadelake ^zlib@1.2.11%gcc@10.2.0 cflags="-O2 -march=native" cxxflags="-O2 -march=native" fflags="-O2 -march=native" +optimize+pic+shared arch=linux-rocky8-cascadelake is unsatisfiable, conflicts are:
#  A conflict was triggered
#  condition(4417)
#  condition(4462)
#  condition(4463)
#  condition(4579)
#  conflict("ucx",4462,4463)
#  dependency_condition(4579,"ucx","cuda")
#  dependency_type(4579,"link")
#  node_compiler_version_satisfies("ucx","gcc","9:","10.2.0")
#  node_compiler_version_set("ucx","gcc","10.2.0")
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
#  root("ucx")
#  variant_condition(4417,"ucx","cuda")
#  variant_set("ucx","cuda","True")
#  version("cuda","10.2.89")
#  version_satisfies("cuda",":10.2.89","10.2.89")

# Note from AMD: +cm option may not work with RHEL 8 and later. 
# https://developer.amd.com/spack/hpc-applications-openmpi/

declare -xr SPACK_PACKAGE='ucx@1.10.1'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='~assertions ~cm +cma +cuda cuda_arch=70,80 +dc ~debug +dm +gdrcopy +ib-hw-tm ~java ~knem ~logging +mlx5-dv +optimizations ~parameter_checking +pic +rc ~rocm +thread_multiple +ud ~xpmem'
declare -xr SPACK_DEPENDENCIES="^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % ${SPACK_COMPILER})"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers  
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

spack spec --long --namespaces --types ucx@1.10.1 % gcc@10.2.0 ~assertions ~cm +cma +cuda cuda_arch=70,80 +dc ~debug +dm +gdrcopy +ib-hw-tm ~java ~knem ~logging +mlx5-dv +optimizations ~parameter_checking +pic +rc ~rocm +thread_multiple +ud ~xpmem "^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all ucx@1.10.1 % gcc@10.2.0 ~assertions ~cm +cma +cuda cuda_arch=70,80 +dc ~debug +dm +gdrcopy +ib-hw-tm ~java ~knem ~logging +mlx5-dv +optimizations ~parameter_checking +pic +rc ~rocm +thread_multiple +ud ~xpmem "^cuda@11.2.2/$(spack find --format '{hash:7}' cuda@11.2.2 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

spack module lmod refresh --delete-tree -y

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'openmpi@4.1.3.sh'

sleep 60
