#!/usr/bin/env bash

#SBATCH --job-name=lammps@20210310
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
declare -xr COMPILER_MODULE='gcc/10.2.0'
declare -xr MPI_MODULE='openmpi/4.1.3'
declare -xr CUDA_MODULE='cuda/11.2.2'

echo "${UNIX_TIME} ${SLURM_JOB_ID} ${SLURM_JOB_MD5SUM} ${SLURM_JOB_DEPENDENCY}" 
echo ""

cat "${SLURM_JOB_SCRIPT}"

module purge
module load "${SCHEDULER_MODULE}"
. "${SPACK_INSTANCE_DIR}/share/spack/setup-env.sh"
module use "${SPACK_ROOT}/share/spack/lmod/linux-rocky8-x86_64/Core"
module load "${COMPILER_MODULE}"
module load "${MPI_MODULE}"
module load "${CUDA_MODULE}"
module list

# A conflict was triggered
#  condition(2667)
#  condition(2883)
#  condition(2884)
#  conflict("lammps",2883,2884)
#  no version satisfies the given constraints
#  root("lammps")
#  variant_condition(2667,"lammps","meam")
#  variant_set("lammps","meam","True")
#  version_satisfies("lammps","20181212:","20210310")
#  version_satisfies("lammps","20210310")

#1 error found in build log:
#     327    -- <<< FFT settings >>>
#     328    -- Primary FFT lib:  FFTW3
#     329    -- Using double precision FFTs
#     330    -- Using non-threaded FFTs
#     331    -- Kokkos FFT: cuFFT
#     332    -- Configuring done
#  >> 333    CMake Error: The following variables are used in this project, but 
#            they are set to NOTFOUND.
#     334    Please set them or make sure they are set and tested correctly in t
#            he CMake files:
#     335    CUDA_CUDA_LIBRARY (ADVANCED)
#     336        linked by target "nvc_get_devices" in directory /tmp/mkandes/sp
#            ack-stage/spack-stage-lammps-20210310-il34zpxttkmdye5sk2shvxlxolpi6
#            tux/spack-src/cmake
#     337        linked by target "gpu" in directory /tmp/mkandes/spack-stage/sp
#            ack-stage-lammps-20210310-il34zpxttkmdye5sk2shvxlxolpi6tux/spack-sr
#            c/cmake
#     338    
#     339    -- Generating done

# FIX: https://github.com/floydhub/dl-docker/pull/48
declare -xr CUDA_CUDA_LIBRARY='/cm/local/apps/cuda/libs/current/lib64'
declare -xr CMAKE_LIBRARY_PATH="${CUDA_CUDA_LIBRARY}"

# >> 6059    /home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rock
#             y8-cascadelake/gcc-10.2.0/kokkos-3.4.01-hkmc634lei4z23r7tvrhaag3ho
#             wgnixn/include/Cuda/Kokkos_Cuda_Parallel.hpp(464): error: calling 
#             a __host__ function("LAMMPS_NS::MinKokkos::force_clear()::[lambda(
#             int) (instance 1)]::operator ()(int) const") from a __device__ fun
#             ction("Kokkos::Impl::ParallelFor< ::LAMMPS_NS::MinKokkos::force_cl
#             ear()   ::[lambda(int) (instance 1)],  ::Kokkos::RangePolicy< ::Ko
#             kkos::Cuda > ,  ::Kokkos::Cuda> ::exec_range<void>  const") is not
#              allowed

#  condition(2731)
#  condition(2995)
#  condition(5719)
#  dependency_condition(2995,"lammps","python")
#  dependency_type(2995,"link")
#  hash("plumed","n2o2udniskgvoaacgn66fbladjkjtcai")
#  imposed_constraint("n2o2udniskgvoaacgn66fbladjkjtcai","hash","python","uasyy5n4yauliglzcgk27zmfa3ltehdy")
#  root("lammps")
#  variant_condition(2731,"lammps","python")
#  variant_condition(5719,"python","optimizations")
#  variant_set("lammps","python","True")
#  variant_set("python","optimizations","False")

# 2 errors found in build log:
#     53    -- Found CURL: /usr/lib64/libcurl.so (found version "7.61.1")
#     54    -- Checking for module 'libzstd>=1.4'
#     55    --   Found libzstd, version 1.4.4
#     56    -- Looking for C++ include cmath
#     57    -- Looking for C++ include cmath - found
#     58    -- Checking external potential C_10_10.mesocnt from https://download
#           .lammps.org/potentials
#  >> 59    CMake Error at Modules/LAMMPSUtils.cmake:101 (file):
#     60      file DOWNLOAD HASH mismatch
#     61    
#     62        for file: [/tmp/mkandes/spack-stage/spack-stage-lammps-20210310-
#           oi3mpq35ufuhx6mnichykaxhdhdrfhf4/spack-build-oi3mpq3/C_10_10.mesocnt
#           ]
#     63          expected hash: [028de73ec828b7830d762702eda571c1]
#     64            actual hash: [d41d8cd98f00b204e9800998ecf8427e]
#     65                 status: [6;"Couldn't resolve host name"]
#     66    
#     67    Call Stack (most recent call first):
#     68      CMakeLists.txt:428 (FetchPotentials)
#     69    
#     70    
#     71    -- Checking external potential TABTP_10_10.mesont from https://downl
#           oad.lammps.org/potentials
#  >> 72    CMake Error at Modules/LAMMPSUtils.cmake:101 (file):
#     73      file DOWNLOAD HASH mismatch
#     74    
#     75        for file: [/tmp/mkandes/spack-stage/spack-stage-lammps-20210310-
#           oi3mpq35ufuhx6mnichykaxhdhdrfhf4/spack-build-oi3mpq3/TABTP_10_10.mes
#           ont]
#     76          expected hash: [744a739da49ad5e78492c1fc9fd9f8c1]
#     77            actual hash: [d41d8cd98f00b204e9800998ecf8427e]
#     78                 status: [6;"Couldn't resolve host name"]

# >> 6059    /home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rock
#             y8-cascadelake/gcc-10.2.0/kokkos-3.4.01-owdzetzbm2fsbcpt3vdiv2lvvl
#             pxqwo7/include/Cuda/Kokkos_Cuda_Parallel.hpp(488): error: calling 
#             a __host__ function("LAMMPS_NS::MinKokkos::force_clear()::[lambda(
#             int) (instance 1)]::operator () const") from a __device__ function
#             ("Kokkos::Impl::ParallelFor< ::LAMMPS_NS::MinKokkos::force_clear()
#                ::[lambda(int) (instance 1)],  ::Kokkos::RangePolicy< ::Kokkos:
#             :Cuda > ,  ::Kokkos::Cuda> ::operator () const") is not allowed
#     6060    
#  >> 6061    /home/mkandes/cm/shared/apps/spack/0.17.3/gpu/opt/spack/linux-rock
#             y8-cascadelake/gcc-10.2.0/kokkos-3.4.01-owdzetzbm2fsbcpt3vdiv2lvvl
#             pxqwo7/include/Cuda/Kokkos_Cuda_Parallel.hpp(488): error: identifi
#             er "LAMMPS_NS::MinKokkos::force_clear()::[lambda(int) (instance 1)
#             ]::operator () const" is undefined in device code

declare -xr SPACK_PACKAGE='lammps@20210310'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='+asphere +body +class2 +colloid +compress +coreshell +cuda cuda_arch=70 +dipole ~exceptions +ffmpeg +granular ~ipo +jpeg +kim +kokkos +kspace ~latte +lib +manybody +mc ~meam +misc +mliap +molecule +mpi +mpiio ~opencl +openmp +opt +peri +png +poems +python +qeq +replica +rigid +shock +snap +spin +srd ~user-adios +user-atc +user-awpmd +user-bocs +user-cgsdk +user-colvars +user-diffraction +user-dpd +user-drude +user-eff +user-fep ~user-h5md +user-lb +user-manifold +user-meamc +user-mesodpd +user-mesont +user-mgpt +user-misc +user-mofff ~user-netcdf ~user-omp +user-phonon +user-plumed +user-ptm +user-qtb +user-reaction +user-reaxc +user-sdpd +user-smd +user-smtbq +user-sph +user-tally +user-uef +user-yaff +voronoi'
declare -xr SPACK_DEPENDENCIES="^openblas@0.3.18/$(spack find --format '{hash:7}' openblas@0.3.18 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^kokkos@3.4.01/$(spack find --format '{hash:7}' kokkos@3.4.01 % ${SPACK_COMPILER} ^kokkos-nvcc-wrapper ~mpi) ^ffmpeg@4.3.2/$(spack find --format '{hash:7}' ffmpeg@4.3.2 % ${SPACK_COMPILER}) ^plumed@2.6.3/$(spack find --format '{hash:7}' plumed@2.6.3 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^python@3.8.12/$(spack find --format '{hash:7}' python@3.8.12 % ${SPACK_COMPILER})"
declare -xr SPACK_SPEC="${SPACK_PACKAGE} % ${SPACK_COMPILER} ${SPACK_VARIANTS} ${SPACK_DEPENDENCIES}"

printenv

spack config get compilers
spack config get config  
spack config get mirrors
spack config get modules
spack config get packages
spack config get repos
spack config get upstreams

spack spec --long --namespaces --types lammps@20210310 % gcc@10.2.0 +asphere +body +class2 +colloid +compress +coreshell +cuda cuda_arch=70 +dipole ~exceptions +ffmpeg +granular ~ipo +jpeg +kim +kokkos +kspace ~latte +lib +manybody +mc ~meam +misc +mliap +molecule +mpi +mpiio ~opencl +openmp +opt +peri +png +poems +python +qeq +replica +rigid +shock +snap +spin +srd ~user-adios +user-atc +user-awpmd +user-bocs +user-cgsdk +user-colvars +user-diffraction +user-dpd +user-drude +user-eff +user-fep ~user-h5md +user-lb +user-manifold +user-meamc +user-mesodpd +user-mesont +user-mgpt +user-misc +user-mofff ~user-netcdf ~user-omp +user-phonon +user-plumed +user-ptm +user-qtb +user-reaction +user-reaxc +user-sdpd +user-smd +user-smtbq +user-sph +user-tally +user-uef +user-yaff +voronoi "^openblas@0.3.18/$(spack find --format '{hash:7}' openblas@0.3.18 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^kokkos@3.4.01/$(spack find --format '{hash:7}' kokkos@3.4.01 % ${SPACK_COMPILER} ^kokkos-nvcc-wrapper +mpi) ^ffmpeg@4.3.2/$(spack find --format '{hash:7}' ffmpeg@4.3.2 % ${SPACK_COMPILER}) ^plumed@2.6.3/$(spack find --format '{hash:7}' plumed@2.6.3 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^python@3.8.12/$(spack find --format '{hash:7}' python@3.8.12 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack concretization failed.'
  exit 1
fi

time -p spack install --jobs "${SLURM_CPUS_PER_TASK}" --fail-fast --yes-to-all lammps@20210310 % gcc@10.2.0 +asphere +body +class2 +colloid +compress +coreshell +cuda cuda_arch=70 +dipole ~exceptions +ffmpeg +granular ~ipo +jpeg +kim +kokkos +kspace ~latte +lib +manybody +mc ~meam +misc +mliap +molecule +mpi +mpiio ~opencl +openmp +opt +peri +png +poems +python +qeq +replica +rigid +shock +snap +spin +srd ~user-adios +user-atc +user-awpmd +user-bocs +user-cgsdk +user-colvars +user-diffraction +user-dpd +user-drude +user-eff +user-fep ~user-h5md +user-lb +user-manifold +user-meamc +user-mesodpd +user-mesont +user-mgpt +user-misc +user-mofff ~user-netcdf ~user-omp +user-phonon +user-plumed +user-ptm +user-qtb +user-reaction +user-reaxc +user-sdpd +user-smd +user-smtbq +user-sph +user-tally +user-uef +user-yaff +voronoi "^openblas@0.3.18/$(spack find --format '{hash:7}' openblas@0.3.18 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} ~mpi ~openmp) ^ffmpeg@4.3.2/$(spack find --format '{hash:7}' ffmpeg@4.3.2 % ${SPACK_COMPILER}) ^kokkos@3.4.01/$(spack find --format '{hash:7}' kokkos@3.4.01 % ${SPACK_COMPILER} ^kokkos-nvcc-wrapper +mpi) ^plumed@2.6.3/$(spack find --format '{hash:7}' plumed@2.6.3 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^python@3.8.12/$(spack find --format '{hash:7}' python@3.8.12 % ${SPACK_COMPILER})"
if [[ "${?}" -ne 0 ]]; then
  echo 'ERROR: spack install failed.'
  exit 1
fi

spack module lmod refresh --delete-tree -y

#sbatch --dependency="afterok:${SLURM_JOB_ID}" ''

sleep 60
