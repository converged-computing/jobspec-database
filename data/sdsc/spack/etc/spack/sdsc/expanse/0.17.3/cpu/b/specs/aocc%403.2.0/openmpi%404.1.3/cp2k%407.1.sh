#!/usr/bin/env bash

#SBATCH --job-name=cp2k@7.1
#SBATCH --account=use300
#SBATCH --reservation=rocky8u7_testing
#SBATCH --partition=ind-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=%x.o%j.%N

declare -xr LOCAL_TIME="$(date +'%Y%m%dT%H%M%S%z')"
declare -xir UNIX_TIME="$(date +'%s')"

declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"
declare -xr TMPDIR="${LOCAL_SCRATCH_DIR}"

declare -xr SYSTEM_NAME='expanse'

declare -xr SPACK_VERSION='0.17.3'
declare -xr SPACK_INSTANCE_NAME='cpu'
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

# unsatisfiable, conflicts are:
#  condition(1295)
#  condition(672)
#  condition(678)
#  condition(891)
#  condition(893)
#  condition(921)
#  condition(923)
#  dependency_condition(891,"cp2k","elpa")
#  dependency_condition(921,"cp2k","python")
#  dependency_type(1295,"build")
#  dependency_type(891,"build")
#  dependency_type(921,"build")
#  imposed_constraint(1295,"version_satisfies","python",":2")
#  imposed_constraint(893,"version_satisfies","elpa","2018.05:2020.11.001")
#  imposed_constraint(923,"version_satisfies","python","3:")
#  no version satisfies the given constraints
#  root("cp2k")
#  variant_condition(672,"cp2k","elpa")
#  variant_condition(678,"cp2k","openmp")
#  variant_set("cp2k","elpa","True")
#  variant_set("cp2k","openmp","True")
#  version_satisfies("cp2k","7.0:8.2","8.2")
#  version_satisfies("cp2k","8.2")
#  version_satisfies("cp2k","8:","8.2")
#  version_satisfies("elpa",":2020.05.001","2018.05.001.rc1")
#  version_satisfies("elpa",":2020.05.001","2018.11.001")
#  version_satisfies("elpa",":2020.05.001","2019.05.002")
#  version_satisfies("elpa",":2020.05.001","2019.11.001")
#  version_satisfies("elpa",":2020.05.001","2020.05.001")

# unsatisfiable, conflicts are:
#  condition(5113)
#  condition(678)
#  condition(911)
#  dependency_type(911,"build")
#  hash("cosma","q76zrnnttgznpfxhjopb5fisi34ay4sd")
#  imposed_constraint("fxzqxj3ljgy5sox5pq7e4sjtqrt75pqf","node","openblas")
#  imposed_constraint("q76zrnnttgznpfxhjopb5fisi34ay4sd","hash","openblas","fxzqxj3ljgy5sox5pq7e4sjtqrt75pqf")
#  imposed_constraint(911,"variant_set","openblas","threads","openmp")
#  root("cp2k")
#  variant_condition(5113,"openblas","threads")
#  variant_condition(678,"cp2k","openmp")
#  variant_set("cp2k","openmp","True")

# unsatisfiable, conflicts are:
#  A conflict was triggered
#  All dependencies must be reachable from root
#  condition(678)
#  condition(842)
#  condition(843)
#  conflict("cp2k",842,843)
#  no version satisfies the given constraints
#  node("fftw")
#  variant_condition(678,"cp2k","openmp")
#  variant_set("cp2k","openmp","False")
#  version_satisfies("cp2k","8.2")
#  version_satisfies("cp2k","8:","8.2")

# unsatisfiable, conflicts are:
#  A conflict was triggered
#  All dependencies must be reachable from root
#  condition(674)
#  condition(831)
#  condition(833)
#  conflict("cp2k",831,833)
#  no version satisfies the given constraints
#  node("fftw")
#  variant_condition(674,"cp2k","libvori")
#  variant_set("cp2k","libvori","True")
#  version_satisfies("cp2k","7.1")
#  version_satisfies("cp2k",":7","7.1")

# unsatisfiable, conflicts are:
#  A conflict was triggered
#  All dependencies must be reachable from root
#  condition(666)
#  condition(831)
#  condition(832)
#  conflict("cp2k",831,832)
#  no version satisfies the given constraints
#  node("fftw")
#  variant_condition(666,"cp2k","cosma")
#  variant_set("cp2k","cosma","True")
#  version_satisfies("cp2k","7.1")
#  version_satisfies("cp2k",":7","7.1")

# ==> Error: ProcessError: Command exited with status 2:
#    'make' '-j16' 'CC=/home/mkandes/cm/shared/apps/spack/0.17.2/cpu/lib/spack/env/gcc/gcc' 'CXX=/home/mkandes/cm/shared/apps/spack/0.17.2/cpu/lib/spack/env/gcc/g++'
#'FC=/home/mkandes/cm/shared/apps/spack/0.17.2/cpu/lib/spack/env/gcc/gfortran' 'PREFIX=/home/mkandes/cm/shared/apps/spack/0.17.2/cpu/opt/spack/linux-rocky8-zen2/gcc-10.2.0/libxsmm-1.16.3-iui4fgda4daxdwqh3tvx52yw7u4ttliy' 'SYM=1'
#2423 errors found in build log:
#     101     ar: creating lib/libxsmmnoblas.a
#     102     ar -rs lib/libxsmmgen.a obj/intel64/generator_spgemm_csc_bsparse.o
#              obj/intel64/generator_matcopy_avx_avx512.o obj/intel64/generator_
#             gemm_common.o obj/intel64/generator_packed_getrf_avx_avx512.o obj/
#             intel64/generator_spgemm_csc_bsparse_soa.o obj/intel64/generator_g
#             emm_sse3_avx_avx2_avx512.o obj/intel64/generator_gemm_avx512_micro
#             kernel.o obj/intel64/generator_gemm_sse3_microkernel.o obj/intel64
#             /generator_mateltwise_avx_avx512.o obj/intel64/generator_packed.o 
#             obj/intel64/generator_spgemm_csc_reader.o obj/intel64/generator_sp
#             gemm_csr_asparse_reg.o obj/intel64/generator_gemm_avx_microkernel.
#             o obj/intel64/generator_packed_trmm_avx_avx512.o obj/intel64/gener
#             ator_matcopy.o obj/intel64/generator_mateltwise.o obj/intel64/gene
#             rator_spgemm_csr_reader.o obj/intel64/generator_spgemm_csc_csparse
#             _soa.o obj/intel64/generator_spgemm_csr_asparse.o obj/intel64/gene
#             rator_gemm_avx2_microkernel.o obj/intel64/generator_common.o obj/i
#             ntel64/generator_x86_instructions.o obj/intel64/generator_packed_g
#             emm_ac_rm_avx_avx2_avx512.o obj/intel64/generator_spgemm_csr_aspar
#             se_soa.o obj/intel64/generator_transpose_avx_avx512.o obj/intel64/
#             generator_packed_gemm_avx_avx512.o obj/intel64/generator_spgemm_cs
#             c_asparse.o obj/intel64/generator_packed_trsm_avx_avx512.o obj/int
#             el64/generator_packed_gemm_bc_rm_avx_avx2_avx512.o obj/intel64/gen
#             erator_gemm.o obj/intel64/generator_transpose.o obj/intel64/genera
#             tor_gemm_noarch.o obj/intel64/generator_spgemm_csr_bsparse_soa.o o
#             bj/intel64/generator_spgemm.o obj/intel64/libxsmm_cpuid_x86.o obj/
#             intel64/libxsmm_generator.o obj/intel64/libxsmm_trace.o
#     103     ar: creating lib/libxsmmgen.a
#     104     /home/mkandes/cm/shared/apps/spack/0.17.2/cpu/lib/spack/env/gcc/gc
#             c -dynamic -o bin/libxsmm_gemm_generator obj/intel64/libxsmm_gener
#             ator_gemm_driver.o -L/tmp/mkandes/spack-stage/spack-stage-libxsmm-
#             1.16.3-iui4fgda4daxdwqh3tvx52yw7u4ttliy/spack-src/lib/ -lxsmmgen \
#     105        -Wl,--gc-sections -Wl,-z,relro,-z,now -Wl,--export-dynamic -lm -l
#             rt -ldl -pthread
#     106     /tmp/ccybASOd.s: Assembler messages:
#  >> 107     /tmp/ccybASOd.s:93414: Error: no such instruction: `vcvtne2ps2bf16
#              %zmm4,%zmm12,%zmm13'

# unsatisfiable, conflicts are:
#  condition(46)
#  condition(732)
#  condition(917)
#  dependency_type(917,"link")
#  hash("amdfftw","t757st4syhikznzwhodz43w232mcjpns")
#  imposed_constraint(917,"variant_set","amdfftw","openmp","True")
#  node("amdfftw")
#  root("cp2k")
#  variant_condition(46,"amdfftw","openmp")
#  variant_condition(732,"cp2k","openmp")
#  variant_set("cp2k","openmp","True")

# unsatisfiable, conflicts are:
#  condition(1349)
#  condition(726)
#  condition(732)
#  condition(945)
#  condition(947)
#  dependency_condition(1349,"elpa","python")
#  dependency_condition(945,"cp2k","elpa")
#  dependency_type(1349,"build")
#  dependency_type(945,"build")
#  hash("amdblis","hx3awsc47q3rvseitz2exudsvxnvt6qa")
#  imposed_constraint("hx3awsc47q3rvseitz2exudsvxnvt6qa","hash","python","kiytcz3y2paxpfhkyelowqdhznp7w2rf")
#  imposed_constraint("kiytcz3y2paxpfhkyelowqdhznp7w2rf","version","python","3.8.12")
#  imposed_constraint(1349,"version_satisfies","python",":2")
#  imposed_constraint(947,"version_satisfies","elpa","2018.05:2020.11.001")
#  no version satisfies the given constraints
#  root("cp2k")
#  variant_condition(726,"cp2k","elpa")
#  variant_condition(732,"cp2k","openmp")
#  variant_set("cp2k","elpa","True")
#  variant_set("cp2k","openmp","True")
#  version_satisfies("cp2k","7.0:8.2","7.1")
#  version_satisfies("cp2k","7.1")
#  version_satisfies("elpa",":2020.05.001","2018.05.001.rc1")
#  version_satisfies("elpa",":2020.05.001","2018.11.001")
#  version_satisfies("elpa",":2020.05.001","2019.05.002")
#  version_satisfies("elpa",":2020.05.001","2019.11.001")
#  version_satisfies("elpa",":2020.05.001","2020.05.001")

declare -xr SPACK_PACKAGE='cp2k@7.1'
declare -xr SPACK_COMPILER='aocc@3.2.0'
declare -xr SPACK_VARIANTS='~cosma ~cuda ~cuda_blas ~cuda_fft ~elpa +libint ~libvori +libxc +mpi +openmp ~pexsi ~plumed ~sirius ~spglib'
declare -xr SPACK_DEPENDENCIES="^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi ~python) ^amdlibflame@3.1/$(spack find --format '{hash:7}' amdlibflame@3.1 % ${SPACK_COMPILER} ~ilp64 threads=none ^amdblis@3.1 ~ilp64 threads=none) ^amdfftw@3.1/$(spack find --format '{hash:7}' amdfftw@3.1 % ${SPACK_COMPILER} ~mpi +openmp) ^amdscalapack@3.1/$(spack find --format '{hash:7}' amdscalapack@3.1 % ${SPACK_COMPILER} ^amdblis@3.1 ~ilp64 threads=none ^openmpi@4.1.3) ^libint@2.6.0 ^libxsmm@1.15 ^libxc@4.2.3"
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

##spack module lmod refresh --delete-tree -y

#sbatch --dependency="afterok:${SLURM_JOB_ID}" 'quantum-espresso@7.0.sh'
sbatch --dependency="afterok:${SLURM_JOB_ID}" 'openfoam@2106.sh'

sleep 30
