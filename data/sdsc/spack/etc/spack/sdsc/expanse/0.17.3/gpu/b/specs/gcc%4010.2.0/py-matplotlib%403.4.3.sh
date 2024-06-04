#!/usr/bin/env bash

#SBATCH --job-name=py-matplotlib@3.4.3
#SBATCH --account=use300
#SBATCH --reservation=rocky8u7_testing
#SBATCH --partition=ind-gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=93G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
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


# new problem on setting +latex
# ==> Error: ProcessError: Command exited with status 1:
#    '/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-rocky8-zen2/gcc-10.2.0/perl-5.32.0-aqkfsadolq2xm6uar2cjxlgordnxp3sm/bin/perl' './install-tl' '-scheme' 'small' '-repository' 'https://ctan.math.washington.edu/tex-archive/systems/texlive/tlnet/' '-portable' '-profile' '/tmp/tmp7qjdb59f'
#See build log for details:
#  /tmp/mkandes/spack-stage/spack-stage-texlive-live-wy2hv3rayfzv4mhw22mhw26h3tzpdask/spack-build-out.txt
declare -xr SPACK_PACKAGE='py-matplotlib@3.4.3'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='+animation +fonts +image ~latex +movies' 
declare -xr SPACK_DEPENDENCIES="^py-numpy@1.20.3/$(spack find --format '{hash:7}' py-numpy@1.20.3 % ${SPACK_COMPILER}) ^openblas@0.3.18/$(spack find --format '{hash:7}' openblas@0.3.18 % ${SPACK_COMPILER} ~ilp64 threads=none) ^imagemagick@7.0.8-7/$(spack find --format '{hash:7}' imagemagick@7.0.8-7 % ${SPACK_COMPILER}) ^ffmpeg@4.3.2/$(spack find --format '{hash:7}' ffmpeg@4.3.2 % ${SPACK_COMPILER})"
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

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'py-scikit-optimize@0.5.2.sh'

sleep 30
