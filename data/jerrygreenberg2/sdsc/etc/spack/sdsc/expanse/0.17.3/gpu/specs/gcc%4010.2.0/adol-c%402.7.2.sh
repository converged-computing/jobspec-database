#!/usr/bin/env bash

#SBATCH --job-name=adol-c@2.7.2
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

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
module list
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"

declare -xr SPACK_PACKAGE='adol-c@2.7.2'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='+advanced_branching +atrig_erf +boost ~doc +examples ~openmp ~sparse'
# Problem installing with +sparse; try and resolve another time.
#==> Installing adol-c
#==> No binary for adol-c found: installing from source
#==> Using cached archive: /home/mkandes/cm/shared/apps/spack/0.17.3/cpu/var/spack/cache/_source-cache/archive/70/701e0856baae91b98397960d5e0a87a549988de9d4002d0e9a56fa08f5455f6e.tar.gz
#==> adol-c: Executing phase: 'autoreconf'
#==> adol-c: Executing phase: 'configure'
#==> adol-c: Executing phase: 'build'
#==> adol-c: Executing phase: 'install'
#==> Error: ProcessError: Command exited with status 2:
#    'make' '-j16' 'install'
#
#2 errors found in build log:
#     974    
#     975    creating build
#     976    creating build/temp.linux-x86_64-3.8
#     977    compile options: '-I/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/o
#            pt/spack/linux-centos8-zen2/gcc-10.2.0/py-numpy-1.19.2-nshfyv5cjnrd
#            ipcszpbwcgnksaqwojih/lib/python3.8/site-packages/numpy/core/include
#             -I/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-ce
#            ntos8-zen2/gcc-10.2.0/python-3.8.5-pwkt76p3o2wk2ay3bv27szze2pnhbsbr
#            /include/python3.8 -I/home/mkandes/adolc_base/include -c'
#     978    extra options: '-std=c++11 -fPIC -w'
#     979    gcc: adolc-python_wrap.cpp
#  >> 980    adolc-python_wrap.cpp:3615:10: fatal error: adolc/adolc.h: No such 
#            file or directory
#     981     3615 | #include <adolc/adolc.h>
#     982          |          ^~~~~~~~~~~~~~~
#     983    compilation terminated.
#  >> 984    error: Command "/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/lib/s
#            pack/env/gcc/gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwr
#            apv -O3 -Wall -fPIC -fPIC -I/home/mkandes/cm/shared/apps/spack/0.15
#            .4/cpu/opt/spack/linux-centos8-zen2/gcc-10.2.0/py-numpy-1.19.2-nshf
#            yv5cjnrdipcszpbwcgnksaqwojih/lib/python3.8/site-packages/numpy/core
#            /include -I/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/
#            linux-centos8-zen2/gcc-10.2.0/python-3.8.5-pwkt76p3o2wk2ay3bv27szze
#            2pnhbsbr/include/python3.8 -I/home/mkandes/adolc_base/include -c ad
#            olc-python_wrap.cpp -o build/temp.linux-x86_64-3.8/adolc-python_wra
#            p.o -std=c++11 -fPIC -w" failed with exit status 1
#     985    make[3]: *** [Makefile:490: install] Error 1
#     986    make[3]: Leaving directory '/tmp/mkandes/spack-stage/spack-stage-ad
#            ol-c-2.7.2-yxhqaj7lcm3yy2ne5womoypirginxaq4/spack-src/ADOL-C/swig'
#     987    make[2]: *** [Makefile:554: install-recursive] Error 1
#     988    make[2]: Leaving directory '/tmp/mkandes/spack-stage/spack-stage-ad
#            ol-c-2.7.2-yxhqaj7lcm3yy2ne5womoypirginxaq4/spack-src/ADOL-C'
#     989    make[1]: *** [Makefile:711: install] Error 2
#     990    make[1]: Leaving directory '/tmp/mkandes/spack-stage/spack-stage-ad
#            ol-c-2.7.2-yxhqaj7lcm3yy2ne5womoypirginxaq4/spack-src/ADOL-C'
#
#See build log for details:
#  /tmp/mkandes/spack-stage/spack-stage-adol-c-2.7.2-yxhqaj7lcm3yy2ne5womoypirginxaq4/spack-build-out.txt
#
#swig -python -c++ -dirvtable -o adolc-python_wrap.cpp adolc-python.i
#../include/adolc/adolc_fatalerror.h:25: Warning 401: Nothing known about base class 'std::exception'. Ignored.
#Warning: Can't read registry to find the necessary compiler setting
#Make sure that Python modules winreg, win32api or win32con are installed.
#C compiler: /home/mkandes/cm/shared/apps/spack/0.17.3/cpu/lib/spack/env/gcc/gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -fPIC
#
#creating build
#creating build/temp.linux-x86_64-3.8
#compile options: '-I/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-centos8-zen2/gcc-10.2.0/py-numpy-1.19.2-nshfyv5cjnrdipcszpbwcgnksaqwojih/lib/python
#3.8/site-packages/numpy/core/include -I/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-centos8-zen2/gcc-10.2.0/python-3.8.5-pwkt76p3o2wk2ay3bv27szze2pn
#hbsbr/include/python3.8 -I/home/mkandes/adolc_base/include -c'
#extra options: '-std=c++11 -fPIC -w'
#gcc: adolc-python_wrap.cpp
#adolc-python_wrap.cpp:3615:10: fatal error: adolc/adolc.h: No such file or directory
# 3615 | #include <adolc/adolc.h>
#      |          ^~~~~~~~~~~~~~~
#compilation terminated.
#error: Command "/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/lib/spack/env/gcc/gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -fPIC -I/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-centos8-zen2/gcc-10.2.0/py-numpy-1.19.2-nshfyv5cjnrdipcszpbwcgnksaqwojih/lib/python3.8/site-packages/numpy/core/include -I/home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-centos8-zen2/gcc-10.2.0/python-3.8.5-pwkt76p3o2wk2ay3bv27szze2pnhbsbr/include/python3.8 -I/home/mkandes/adolc_base/include -c adolc-python_wrap.cpp -o build/temp.linux-x86_64-3.8/adolc-python_wrap.o -std=c++11 -fPIC -w" failed with exit status 1
#make[3]: *** [Makefile:490: install] Error 1
#make[3]: Leaving directory '/tmp/mkandes/spack-stage/spack-stage-adol-c-2.7.2-yxhqaj7lcm3yy2ne5womoypirginxaq4/spack-src/ADOL-C/swig'
#make[2]: *** [Makefile:554: install-recursive] Error 1
#make[2]: Leaving directory '/tmp/mkandes/spack-stage/spack-stage-adol-c-2.7.2-yxhqaj7lcm3yy2ne5womoypirginxaq4/spack-src/ADOL-C'
#make[1]: *** [Makefile:711: install] Error 2
#make[1]: Leaving directory '/tmp/mkandes/spack-stage/spack-stage-adol-c-2.7.2-yxhqaj7lcm3yy2ne5womoypirginxaq4/spack-src/ADOL-C'
#make: *** [Makefile:537: install-recursive] Error 1
#[mkandes@login02 cpu]$
declare -xr SPACK_DEPENDENCIES="^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi)"
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

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'hypre@2.23.0.sh'

sleep 60
