#!/usr/bin/env bash
# real 1214.82

#SBATCH --job-name=petsc@3.16.1
##SBATCH --reservation=root_73
#SBATCH --account=use300
#SBATCH --partition=ind-gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=93G
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

# [mkandes@login02 spack]$ spack --show-cores=minimized spec -l petsc@3.16.1 % gcc@10.2.0 ~X ~batch ~cgns ~complex ~cuda ~debug +double ~exodusii +fftw ~giflib +hdf5 ~hpddm ~hwloc +hypre ~int64 ~jpeg ~knl ~libpng ~libyaml ~memkind +metis ~mkl-pardiso ~mmg ~moab ~mpfr +mpi +mumps ~p4est ~parmmg +ptscotch ~random123 ~rocm ~saws +scalapack +shared ~strumpack +suite-sparse +superlu-dist ~tetgen ~trilinos ~valgrind "^python@3.8.12/$(spack find --format '{hash:7}' python@3.8.12 % ${SPACK_COMPILER}) ^openblas@0.3.18/$(spack find --format '{hash:7}' openblas@0.3.18 % ${SPACK_COMPILER} ~ilp64 threads=none) ^fftw@3.3.10/$(spack find --format '{hash:7}' fftw@3.3.10 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.1) ^hdf5@1.10.7/$(spack find --format '{hash:7}' hdf5@1.10.7 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.1) ^hypre@2.23.0/$(spack find --format '{hash:7}' hypre@2.23.0 % ${SPACK_COMPILER} ~int64 +mpi ^openmpi@4.1.1) ^mumps@5.4.0/$(spack find --format '{hash:7}' mumps@5.4.0 % ${SPACK_COMPILER} ~int64 +mpi ^openmpi@4.1.1) ^suite-sparse@5.10.1/$(spack find --format '{hash:7}' suite-sparse@5.10.1 % ${SPACK_COMPILER}) ^superlu-dist@7.1.1/$(spack find --format '{hash:7}' superlu-dist@7.1.1 % ${SPACK_COMPILER} ~int64 ^openmpi@4.1.1)"
#Input spec
#--------------------------------
#petsc@3.16.1%gcc@10.2.0~X~batch~cgns~complex~cuda~debug+double~exodusii+fftw~giflib+hdf5~hpddm~hwloc+hypre~int64~jpeg~knl~libpng~libyaml~memkind+metis~mkl-pardiso~mmg~moab~mpfr+mpi+mumps~p4est~parmmg+ptscotch~random123~rocm~saws+scalapack+shared~strumpack+suite-sparse+superlu-dist~tetgen~trilinos~valgrind
#    ^fftw@3.3.10%gcc@10.2.0+mpi~openmp~pfft_patches precision=double,float arch=linux-rocky8-zen2
#        ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
#            ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2
#                ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                    ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                    ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                    ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#            ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2
#                ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#            ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#            ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2
#            ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2
#            ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2
#                ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#    ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#        ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
#            ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2
#                ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                    ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                    ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                    ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#            ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2
#                ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#            ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2
#            ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2
#            ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2
#                ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^hypre@2.23.0%gcc@10.2.0~complex~cuda~debug+fortran~int64~internal-superlu~mixedint+mpi~openmp+shared+superlu-dist~unified-memory cuda_arch=none arch=linux-rocky8-zen2
#        ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#        ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
#            ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2
#                ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                    ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                    ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                    ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#            ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2
#                ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#            ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#            ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2
#            ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2
#            ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2
#                ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#            ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#            ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#    ^mumps@5.4.0%gcc@10.2.0~blr_mt+complex+double+float~int64+metis+mpi+openmp+parmetis+ptscotch+scotch+shared patches=1946864d2106f7414aaa4dbd4dbc068b7804af7c1588381e814b268a56140a52 arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^netlib-scalapack@2.1.0%gcc@10.2.0~ipo+pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-rocky8-zen2
#            ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#            ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
#                ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2
#                    ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                        ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                        ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                        ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                    ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#                ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2
#                    ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#                ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#                ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2
#                ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2
#                ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2
#                    ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#        ^scotch@6.1.1%gcc@10.2.0+compression+esmumps~int64~metis+mpi+shared arch=linux-rocky8-zen2
#    ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#    ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2
#        ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2
#        ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2
#            ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#        ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2
#            ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#            ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#            ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2
#        ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#        ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2
#        ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^suite-sparse@5.10.1%gcc@10.2.0~cuda~openmp+pic~tbb arch=linux-rocky8-zen2
#        ^gmp@6.2.1%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^mpfr@4.1.0%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#    ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#        ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
#            ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2
#                ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                    ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                    ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                    ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#            ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2
#                ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#            ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#            ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2
#            ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2
#            ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2
#                ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#
#Concretized
#--------------------------------
#==> Error: petsc@3.16.1%gcc@10.2.0~X~batch~cgns~complex~cuda~debug+double~exodusii+fftw~giflib+hdf5~hpddm~hwloc+hypre~int64~jpeg~knl~libpng~libyaml~memkind+metis~mkl-pardiso~mmg~moab~mpfr+mpi+mumps~p4est~parmmg+ptscotch~random123~rocm~saws+scalapack+shared~strumpack+suite-sparse+superlu-dist~tetgen~trilinos~valgrind ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2 ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2 ^fftw@3.3.10%gcc@10.2.0+mpi~openmp~pfft_patches precision=double,float arch=linux-rocky8-zen2 ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2 ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2 ^gmp@6.2.1%gcc@10.2.0 arch=linux-rocky8-zen2 ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hypre@2.23.0%gcc@10.2.0~complex~cuda~debug+fortran~int64~internal-superlu~mixedint+mpi~openmp+shared+superlu-dist~unified-memory cuda_arch=none arch=linux-rocky8-zen2 ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^mpfr@4.1.0%gcc@10.2.0 arch=linux-rocky8-zen2 ^mumps@5.4.0%gcc@10.2.0~blr_mt+complex+double+float~int64+metis+mpi+openmp+parmetis+ptscotch+scotch+shared patches=1946864d2106f7414aaa4dbd4dbc068b7804af7c1588381e814b268a56140a52 arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^netlib-scalapack@2.1.0%gcc@10.2.0~ipo+pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openblas@0.3.18%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2 ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.1%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2 ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@28.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2 ^scotch@6.1.1%gcc@10.2.0+compression+esmumps~int64~metis+mpi+shared arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2 ^suite-sparse@5.10.1%gcc@10.2.0~cuda~openmp+pic~tbb arch=linux-rocky8-zen2 ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2 ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2 ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 is unsatisfiable, conflicts are:
#  Single valued variants must have a single value
#  condition(3724)
#  condition(4352)
#  condition(4358)
#  condition(4359)
#  condition(4627)
#  condition(4628)
#  dependency_type(4627,"build")
#  dependency_type(4628,"build")
#  hash("mumps","kleude27zrw73mhrtlgfqyjcssrligze")
#  imposed_constraint(4627,"variant_set","mumps","int64","False")
#  imposed_constraint(4628,"variant_set","mumps","int64","False")
#  node("mumps")
#  root("petsc")
#  variant_condition(3724,"mumps","int64")
#  variant_condition(4352,"petsc","metis")
#  variant_condition(4358,"petsc","mumps")
#  variant_condition(4359,"petsc","openmp")
#  variant_set("petsc","metis","True")
#  variant_set("petsc","mumps","True")
#  variant_single_value("petsc","openmp")

# TESTING: consistencyChecks from config.packages.fftw(/tmp/mkandes/spack-stage/spack-stage-petsc-3.16.1-zpioabo5shpjdb75etokcsav3wq27vg5/spack-src/config/BuildSystem/config/package.py:1002)
# *******************************************************************************
#         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):
# -------------------------------------------------------------------------------
# Cannot use fftw with 64 bit BLAS/Lapack indices
# https://groups.google.com/g/linux.debian.bugs.dist/c/1af2nVr2JTw

# TESTING: consistencyChecks from config.packages.SuperLU_DIST(/tmp/mkandes/spack-stage/spack-stage-petsc-3.16.1-wew4dvfnedahkcvky4j2a2olcey3ihra/spack-src/config/BuildSystem/config/package.py:1002)
#*******************************************************************************
#         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):
#-------------------------------------------------------------------------------
#Cannot use SuperLU_DIST with 64 bit BLAS/Lapack indices
#*******************************************************************************

# TESTING: consistencyChecks from config.packages.SuiteSparse(/tmp/mkandes/spack-stage/spack-stage-petsc-3.16.1-ge7xb23qqhiljw4rsdemrh6xymwthb34/spack-src/config/BuildSystem/config/packages/SuiteSparse.py:176)
#*******************************************************************************
#         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):
#-------------------------------------------------------------------------------
#Cannot use SuiteSparse with 64 bit BLAS/Lapack indices
#*******************************************************************************

# *******************************************************************************
#         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):
#-------------------------------------------------------------------------------
#Cannot use scalapack with 64 bit BLAS/Lapack indices
#*******************************************************************************

# *******************************************************************************
#         UNABLE to CONFIGURE with GIVEN OPTIONS    (see configure.log for details):
#-------------------------------------------------------------------------------
#Cannot use hypre with 64 bit BLAS/Lapack indices
#*******************************************************************************

declare -xr SPACK_PACKAGE='petsc@3.16.1'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='~X ~batch ~cgns ~complex ~cuda ~debug +double ~exodusii ~fftw ~giflib +hdf5 ~hpddm ~hwloc ~hypre ~int64 ~jpeg ~knl ~libpng ~libyaml ~memkind +metis ~mkl-pardiso ~mmg ~moab ~mpfr +mpi ~mumps ~p4est ~parmmg +ptscotch ~random123 ~rocm ~saws ~scalapack +shared ~strumpack ~suite-sparse ~superlu-dist ~tetgen ~trilinos ~valgrind'
declare -xr SPACK_DEPENDENCIES="^python@3.8.12/$(spack find --format '{hash:7}' python@3.8.12 % ${SPACK_COMPILER}) ^openblas@0.3.18/$(spack find --format '{hash:7}' openblas@0.3.18 % ${SPACK_COMPILER} ~ilp64 threads=none) ^hdf5@1.10.7/$(spack find --format '{hash:7}' hdf5@1.10.7 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.1) ^parmetis@4.0.3/$(spack find --format '{hash:7}' parmetis@4.0.3 % ${SPACK_COMPILER} ~int64 ^openmpi@4.1.1)"
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

sbatch --dependency="afterok:${SLURM_JOB_ID}" 'petsc@3.16.1-cmplx.sh'

sleep 60
