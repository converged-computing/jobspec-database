#!/usr/bin/env bash

#SBATCH --job-name=gdb@11.1
#SBATCH --account=sys200
#SBATCH --partition=hotel-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -w gpu1
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

module purge
module load "${SCHEDULER_MODULE}"
module list
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"


#[mkandes@login02 ~]$ spack --show-cores=minimized spec -l gdb@11.1 % gcc@8.5.0 ~gold ~ld ~lto +python ~quad ~source-highlight ~tui +xz "^python@3.8.12/$(spack find --format '{hash:7}' python@3.8.12 % ${SPACK_COMPILER})" 
#Input spec
#--------------------------------
#gdb@11.1%gcc@8.5.0~gold~ld~lto+python~quad~source-highlight~tui+xz
#    ^python@3.8.12%gcc@8.5.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2
#        ^bzip2@1.0.8%gcc@8.5.0~debug~pic+shared arch=linux-rocky8-zen2
#        ^expat@2.4.1%gcc@8.5.0+libbsd arch=linux-rocky8-zen2
#            ^libbsd@0.11.3%gcc@8.5.0 arch=linux-rocky8-zen2
#                ^libmd@1.0.3%gcc@8.5.0 arch=linux-rocky8-zen2
#        ^gdbm@1.19%gcc@8.5.0 arch=linux-rocky8-zen2
#            ^readline@8.1%gcc@8.5.0 arch=linux-rocky8-zen2
#                ^ncurses@6.2%gcc@8.5.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#        ^gettext@0.21%gcc@8.5.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2
#            ^libiconv@1.16%gcc@8.5.0 libs=shared,static arch=linux-rocky8-zen2
#            ^libxml2@2.9.12%gcc@8.5.0~python arch=linux-rocky8-zen2
#                ^xz@5.2.5%gcc@8.5.0~pic libs=shared,static arch=linux-rocky8-zen2
#                ^zlib@1.2.11%gcc@8.5.0+optimize+pic+shared arch=linux-rocky8-zen2
#            ^tar@1.34%gcc@8.5.0 arch=linux-rocky8-zen2
#        ^libffi@3.3%gcc@8.5.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2
#        ^openssl@1.1.1k%gcc@8.5.0~docs certs=system arch=linux-rocky8-zen2
#        ^sqlite@3.36.0%gcc@8.5.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2
#        ^util-linux-uuid@2.36.2%gcc@8.5.0 arch=linux-rocky8-zen2
#
#Concretized
#--------------------------------
#==> Error: gdb@11.1%gcc@8.5.0~gold~ld~lto+python~quad~source-highlight~tui+xz ^bzip2@1.0.8%gcc@8.5.0~debug~pic+shared arch=linux-rocky8-zen2 ^expat@2.4.1%gcc@8.5.0+libbsd arch=linux-rocky8-zen2 ^gdbm@1.19%gcc@8.5.0 arch=linux-rocky8-zen2 ^gettext@0.21%gcc@8.5.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2 ^libbsd@0.11.3%gcc@8.5.0 arch=linux-rocky8-zen2 ^libffi@3.3%gcc@8.5.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@8.5.0 libs=shared,static arch=linux-rocky8-zen2 ^libmd@1.0.3%gcc@8.5.0 arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@8.5.0~python arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@8.5.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@8.5.0~docs certs=system arch=linux-rocky8-zen2 ^python@3.8.12%gcc@8.5.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2 ^readline@8.1%gcc@8.5.0 arch=linux-rocky8-zen2 ^sqlite@3.36.0%gcc@8.5.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2 ^tar@1.34%gcc@8.5.0 arch=linux-rocky8-zen2 ^util-linux-uuid@2.36.2%gcc@8.5.0 arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@8.5.0~pic libs=shared,static arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@8.5.0+optimize+pic+shared arch=linux-rocky8-zen2 is unsatisfiable, conflicts are:
#  condition(3738)
#  condition(713)
#  condition(724)
#  dependency_condition(724,"gdb","python")
#  dependency_type(724,"run")
#  hash("python","dtdsuje4z5v7rqevinukk4n37p3lxjcn")
#  imposed_constraint(724,"variant_set","python","debug","True")
#  root("gdb")
#  variant_condition(3738,"python","debug")
#  variant_condition(713,"gdb","python")
#  variant_set("gdb","python","True")

declare -xr SPACK_PACKAGE='gdb@11.1'
declare -xr SPACK_COMPILER='gcc@8.5.0'
declare -xr SPACK_VARIANTS='~gold ~ld ~lto ~python ~quad ~source-highlight ~tui +xz'
declare -xr SPACK_DEPENDENCIES=''
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

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'ninja@1.10.2.sh'

sleep 20
