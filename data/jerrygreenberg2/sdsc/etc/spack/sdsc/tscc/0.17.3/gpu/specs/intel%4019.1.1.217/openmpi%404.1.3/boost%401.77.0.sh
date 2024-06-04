#!/usr/bin/env bash

#SBATCH --job-name=boost@1.77.0
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
declare -xr SPACK_INSTANCE_DIR="/cm/shared/apps/spack/${SPACK_VERSION}/${SPACK_INSTANCE_NAME}"

declare -xr SLURM_JOB_SCRIPT="$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')"
declare -xr SLURM_JOB_MD5SUM="$(md5sum ${SLURM_JOB_SCRIPT})"

declare -xr SCHEDULER_MODULE='slurm/expanse/current'

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
module list
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"

#7 errors found in build log:
#     13    
#     14    
#     15    ###
#     16    ###
#     17    
#     18    > icpc -x c++ -std=c++11 -O3 -s -static -DNDEBUG builtins.cpp class.
#           cpp command.cpp compile.cpp constants.cpp cwd.cpp debug.cpp debugger
#           .cpp execcmd.cpp execnt.cpp execunix.cpp filesys.cpp filent.cpp file
#           unix.cpp frames.cpp function.cpp glob.cpp hash.cpp hcache.cpp hdrmac
#           ro.cpp headers.cpp jam_strings.cpp jam.cpp jamgram.cpp lists.cpp mak
#           e.cpp make1.cpp md5.cpp mem.cpp modules.cpp native.cpp object.cpp op
#           tion.cpp output.cpp parse.cpp pathnt.cpp pathsys.cpp pathunix.cpp re
#           gexp.cpp rules.cpp scan.cpp search.cpp startup.cpp subst.cpp sysinfo
#           .cpp timestamp.cpp variable.cpp w32_getreg.cpp modules/order.cpp mod
#           ules/path.cpp modules/property-set.cpp modules/regex.cpp modules/seq
#           uence.cpp modules/set.cpp -o b2
#  >> 19    ld: cannot find -lstdc++
#  >> 20    ld: cannot find -lm
#  >> 21    ld: cannot find -lstdc++
#  >> 22    ld: cannot find -lc
#  >> 23    ld: cannot find -ldl
#  >> 24    ld: cannot find -lc
#     25    > cp b2 bjam
#  >> 26    cp: cannot stat 'b2': No such file or directory
#     27    
#     28    Failed to build B2 build engine
#
#See build log for details:
#  /tmp/mkandes/spack-stage/spack-stage-boost-1.77.0-itjji23pwpspc4xcwqoci3ipyiaf4n3n/spack-build-out.txt
#
#==> Error: Terminating after first install failure: ProcessError: Command exited with status 1:
#    './bootstrap.sh' '--prefix=/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-rocky8-zen2/intel-19.1.3.304/boost-1.77.0-itjji23pwpspc4xcwqoci3ipyiaf4n3n' '--with-toolset=intel-linux' '--with-libraries=math,serialization,date_time,mpi,iostreams,timer,exception,system,thread,log,chrono,filesystem,regex,random,graph,atomic,locale,test,wave,program_options,graph_parallel' '--without-icu'

declare -xr INTEL_LICENSE_FILE='40000@elprado.sdsc.edu:40200@elprado.sdsc.edu'
declare -xr SPACK_PACKAGE='boost@1.77.0'
declare -xr SPACK_COMPILER='intel@19.1.1.217'
declare -xr SPACK_VARIANTS='+atomic +chrono ~clanglibcpp ~container ~context ~coroutine +date_time ~debug +exception ~fiber +filesystem +graph ~icu +iostreams +locale +log +math +mpi +multithreaded ~numpy +pic +program_options ~python +random +regex +serialization +shared +signals ~singlethreaded +system ~taggedlayout +test +thread +timer ~versionedlayout +wave'
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

#spack module lmod refresh --delete-tree -y

#sbatch --dependency="afterok:${SLURM_JOB_ID}" 'superlu-dist@7.1.1.sh'

sleep 60
