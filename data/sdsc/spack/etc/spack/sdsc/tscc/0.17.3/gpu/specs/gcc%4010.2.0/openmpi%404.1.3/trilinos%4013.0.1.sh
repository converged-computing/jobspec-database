#!/usr/bin/env bash

#SBATCH --job-name=trilinos@13.0.1
#SBATCH --account=sys200
#SBATCH --partition=hotel-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -w gpu1
#SBATCH --time=01:00:00
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

# [mkandes@login02 spack]$ spack spec -l trilinos@13.0.1 % gcc@10.2.0 +adios2 +amesos +amesos2 +anasazi +aztec ~basker +belos +boost ~chaco ~complex ~cuda ~cuda_rdc ~debug ~dtk +epetra +epetraext ~epetraextbtf ~epetraextexperimental ~epetraextgraphreorderings ~exodus +explicit_template_instantiation ~float +fortran +gtest +hdf5 +hypre +ifpack +ifpack2 ~intrepid ~intrepid2 ~ipo ~isorropia +kokkos ~mesquite ~minitensor +ml +mpi +muelu +mumps ~nox ~openmp ~phalanx ~piro +python ~rol ~rythmos +sacado ~scorec ~shards +shared ~shylu ~stk ~stokhos ~stratimikos ~strumpack +suite-sparse +superlu +superlu-dist ~teko ~tempus +tpetra ~trilinoscouplings ~wrapper ~x11 +zoltan +zoltan2
#Input spec
#--------------------------------
#trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5+hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse+superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2
#
#Concretized
#--------------------------------
#==> Error: trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5+hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse+superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2 is unsatisfiable, errors are:
#  A conflict was triggered
#
#    To see full clingo unsat cores, re-run with `spack --show-cores=full`
#    For full, subset-minimal unsat cores, re-run with `spack --show-cores=minimized
#    Warning: This may take (up to) hours for some specs
#[mkandes@login02 spack]$ spack --show-cores=minimized spec -l trilinos@13.0.1 % gcc@10.2.0 +adios2 +amesos +amesos2 +anasazi +aztec ~basker +belos +boost ~chaco ~complex ~cuda ~cuda_rdc ~debug ~dtk +epetra +epetraext ~epetraextbtf ~epetraextexperimental ~epetraextgraphreorderings ~exodus +explicit_template_instantiation ~float +fortran +gtest +hdf5 +hypre +ifpack +ifpack2 ~intrepid ~intrepid2 ~ipo ~isorropia +kokkos ~mesquite ~minitensor +ml +mpi +muelu +mumps ~nox ~openmp ~phalanx ~piro +python ~rol ~rythmos +sacado ~scorec ~shards +shared ~shylu ~stk ~stokhos ~stratimikos ~strumpack +suite-sparse +superlu +superlu-dist ~teko ~tempus +tpetra ~trilinoscouplings ~wrapper ~x11 +zoltan +zoltan2
#Input spec
#--------------------------------
#trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5+hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse+superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2
#
#Concretized
#--------------------------------
#==> Error: trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5+hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse+superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2 is unsatisfiable, conflicts are:
#  A conflict was triggered
#  condition(5882)
#  condition(5883)
#  condition(6128)
#  condition(6129)
#  conflict("trilinos",6128,6129)
#  root("trilinos")
#  variant_condition(5882,"trilinos","superlu")
#  variant_condition(5883,"trilinos","superlu-dist")
#  variant_set("trilinos","superlu","True")
#  variant_set("trilinos","superlu-dist","True")
#
#[mkandes@login02 spack]$

# [mkandes@login02 spack]$ spack --show-cores=minimized spec -l trilinos@13.0.1 % gcc@10.2.0 +adios2 +amesos +amesos2 +anasazi +aztec ~basker +belos +boost ~chaco ~complex ~cuda ~cuda_rdc ~debug ~dtk +epetra +epetraext ~epetraextbtf ~epetraextexperimental ~epetraextgraphreorderings ~exodus +explicit_template_instantiation ~float +fortran +gtest +hdf5 +hypre +ifpack +ifpack2 ~intrepid ~intrepid2 ~ipo ~isorropia +kokkos ~mesquite ~minitensor +ml +mpi +muelu +mumps ~nox ~openmp ~phalanx ~piro +python ~rol ~rythmos +sacado ~scorec ~shards +shared ~shylu ~stk ~stokhos ~stratimikos ~strumpack +suite-sparse ~superlu +superlu-dist ~teko ~tempus +tpetra ~trilinoscouplings ~wrapper ~x11 +zoltan +zoltan2 "^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi) ^hdf5@1.10.7/$(spack find --format '{hash:7}' hdf5@1.10.7 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^parmetis@4.0.3/$(spack find --format '{hash:7}' parmetis@4.0.3 % ${SPACK_COMPILER} ^openmpi@4.1.3) ^hypre@2.23.0/$(spack find --format '{hash:7}' hypre@2.23.0 % ${SPACK_COMPILER} ~int64 +mpi ^openmpi@4.1.3)"
#Input spec
#--------------------------------
#trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5+hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse~superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2
#    ^boost@1.77.0%gcc@10.2.0+atomic+chrono~clanglibcpp~container~context~coroutine+date_time~debug+exception~fiber+filesystem+graph~icu+iostreams+locale+log+math~mpi+multithreaded+numpy+pic+program_options+python+random+regex+serialization+shared+signals~singlethreaded+system~taggedlayout+test+thread+timer~versionedlayout+wave cxxstd=98 patches=93f4aad8f88d1437e50d95a2d066390ef3753b99ef5de24f7a46bc083bd6df06,b8569d7d4c3ef0501a39857126a2b0a88519bf256c29f3252a6958916ce82255 visibility=hidden arch=linux-rocky8-zen2
#        ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2
#        ^py-numpy@1.20.3%gcc@10.2.0+blas+lapack patches=873745d7b547857fcfec9cae90b09c133b42a4f0c23b6c2d84cf37e2dd816604 arch=linux-rocky8-zen2
#            ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#            ^py-setuptools@58.2.0%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2
#                    ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2
#                        ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                            ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2
#                        ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2
#                            ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#                    ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2
#                        ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                        ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                            ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                            ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                        ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2
#                    ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#                    ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2
#                    ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#        ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^hypre@2.23.0%gcc@10.2.0~complex~cuda~debug+fortran~int64~internal-superlu~mixedint+mpi~openmp+shared+superlu-dist~unified-memory cuda_arch=none arch=linux-rocky8-zen2
#        ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#            ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#            ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#    ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#
#Concretized
#--------------------------------
#==> Error: trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5+hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse~superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2 ^boost@1.77.0%gcc@10.2.0+atomic+chrono~clanglibcpp~container~context~coroutine+date_time~debug+exception~fiber+filesystem+graph~icu+iostreams+locale+log+math~mpi+multithreaded+numpy+pic+program_options+python+random+regex+serialization+shared+signals~singlethreaded+system~taggedlayout+test+thread+timer~versionedlayout+wave cxxstd=98 patches=93f4aad8f88d1437e50d95a2d066390ef3753b99ef5de24f7a46bc083bd6df06,b8569d7d4c3ef0501a39857126a2b0a88519bf256c29f3252a6958916ce82255 visibility=hidden arch=linux-rocky8-zen2 ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2 ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2 ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2 ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2 ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hypre@2.23.0%gcc@10.2.0~complex~cuda~debug+fortran~int64~internal-superlu~mixedint+mpi~openmp+shared+superlu-dist~unified-memory cuda_arch=none arch=linux-rocky8-zen2 ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2 ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^py-numpy@1.20.3%gcc@10.2.0+blas+lapack patches=873745d7b547857fcfec9cae90b09c133b42a4f0c23b6c2d84cf37e2dd816604 arch=linux-rocky8-zen2 ^py-setuptools@58.2.0%gcc@10.2.0 arch=linux-rocky8-zen2 ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2 ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2 ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2 ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 is unsatisfiable, conflicts are:
 # condition(1632)
 # condition(5851)
 # condition(6190)
 # dependency_condition(6190,"trilinos","hypre")
 # dependency_type(6190,"link")
 # hash("hypre","7xcmtpkwg3eg5vdx25ymchauzwodyccm")
 # imposed_constraint(6190,"variant_set","hypre","int64","False")
 # root("trilinos")
 # variant_condition(1632,"hypre","int64")
 # variant_condition(5851,"trilinos","hypre")
 # variant_set("trilinos","hypre","True")

# [mkandes@login02 spack]$ spack --show-cores=minimized spec -l trilinos@13.0.1 % gcc@10.2.0 +adios2 +amesos +amesos2 +anasazi +aztec ~basker +belos +boost ~chaco ~complex ~cuda ~cuda_rdc ~debug ~dtk +epetra +epetraext ~epetraextbtf ~epetraextexperimental ~epetraextgraphreorderings ~exodus +explicit_template_instantiation ~float +fortran +gtest +hdf5 ~hypre +ifpack +ifpack2 ~intrepid ~intrepid2 ~ipo ~isorropia +kokkos ~mesquite ~minitensor +ml +mpi +muelu +mumps ~nox ~openmp ~phalanx ~piro +python ~rol ~rythmos +sacado ~scorec ~shards +shared ~shylu ~stk ~stokhos ~stratimikos ~strumpack +suite-sparse ~superlu +superlu-dist ~teko ~tempus +tpetra ~trilinoscouplings ~wrapper ~x11 +zoltan +zoltan2 "^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi) ^hdf5@1.10.7/$(spack find --format '{hash:7}' hdf5@1.10.7 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^parmetis@4.0.3/$(spack find --format '{hash:7}' parmetis@4.0.3 % ${SPACK_COMPILER} ^openmpi@4.1.3) ^mumps@5.4.0/$(spack find --format '{hash:7}' mumps@5.4.0 % ${SPACK_COMPILER} ~int64 +mpi ^openmpi@4.1.3) ^superlu-dist@7.1.1/$(spack find --format '{hash:7}' superlu-dist@7.1.1 % ${SPACK_COMPILER} ~int64 ^openmpi@4.1.3)"
#Input spec
#--------------------------------
#trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5~hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse~superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2
#    ^boost@1.77.0%gcc@10.2.0+atomic+chrono~clanglibcpp~container~context~coroutine+date_time~debug+exception~fiber+filesystem+graph~icu+iostreams+locale+log+math~mpi+multithreaded+numpy+pic+program_options+python+random+regex+serialization+shared+signals~singlethreaded+system~taggedlayout+test+thread+timer~versionedlayout+wave cxxstd=98 patches=93f4aad8f88d1437e50d95a2d066390ef3753b99ef5de24f7a46bc083bd6df06,b8569d7d4c3ef0501a39857126a2b0a88519bf256c29f3252a6958916ce82255 visibility=hidden arch=linux-rocky8-zen2
#        ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2
#        ^py-numpy@1.20.3%gcc@10.2.0+blas+lapack patches=873745d7b547857fcfec9cae90b09c133b42a4f0c23b6c2d84cf37e2dd816604 arch=linux-rocky8-zen2
#            ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#            ^py-setuptools@58.2.0%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2
#                    ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2
#                        ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                            ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2
#                        ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2
#                            ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#                    ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2
#                        ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                        ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                            ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                            ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                        ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2
#                    ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#                    ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2
#                    ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#        ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^mumps@5.4.0%gcc@10.2.0~blr_mt+complex+double+float~int64+metis+mpi+openmp+parmetis+ptscotch+scotch+shared patches=1946864d2106f7414aaa4dbd4dbc068b7804af7c1588381e814b268a56140a52 arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^netlib-scalapack@2.1.0%gcc@10.2.0~ipo+pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-rocky8-zen2
#            ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#            ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                    ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#        ^scotch@6.1.1%gcc@10.2.0+compression+esmumps~int64~metis+mpi+shared arch=linux-rocky8-zen2
#    ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#    ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#
#Concretized
#--------------------------------
#==> Error: trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5~hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu+mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse~superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2 ^boost@1.77.0%gcc@10.2.0+atomic+chrono~clanglibcpp~container~context~coroutine+date_time~debug+exception~fiber+filesystem+graph~icu+iostreams+locale+log+math~mpi+multithreaded+numpy+pic+program_options+python+random+regex+serialization+shared+signals~singlethreaded+system~taggedlayout+test+thread+timer~versionedlayout+wave cxxstd=98 patches=93f4aad8f88d1437e50d95a2d066390ef3753b99ef5de24f7a46bc083bd6df06,b8569d7d4c3ef0501a39857126a2b0a88519bf256c29f3252a6958916ce82255 visibility=hidden arch=linux-rocky8-zen2 ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2 ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2 ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2 ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2 ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2 ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2 ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2 ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2 ^mumps@5.4.0%gcc@10.2.0~blr_mt+complex+double+float~int64+metis+mpi+openmp+parmetis+ptscotch+scotch+shared patches=1946864d2106f7414aaa4dbd4dbc068b7804af7c1588381e814b268a56140a52 arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2 ^netlib-scalapack@2.1.0%gcc@10.2.0~ipo+pic+shared build_type=Release patches=1c9ce5fee1451a08c2de3cc87f446aeda0b818ebbce4ad0d980ddf2f2a0b2dc4,f2baedde688ffe4c20943c334f580eb298e04d6f35c86b90a1f4e8cb7ae344a2 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2 ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2 ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2 ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2 ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2 ^py-numpy@1.20.3%gcc@10.2.0+blas+lapack patches=873745d7b547857fcfec9cae90b09c133b42a4f0c23b6c2d84cf37e2dd816604 arch=linux-rocky8-zen2 ^py-setuptools@58.2.0%gcc@10.2.0 arch=linux-rocky8-zen2 ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2 ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2 ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2 ^scotch@6.1.1%gcc@10.2.0+compression+esmumps~int64~metis+mpi+shared arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2 ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2 ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2 ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2 ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2 is unsatisfiable, conflicts are:
#  condition(3362)
#  condition(5864)
#  condition(5866)
#  condition(6201)
#  dependency_condition(6201,"trilinos","mumps")
#  dependency_type(6201,"link")
#  hash("mumps","kleude27zrw73mhrtlgfqyjcssrligze")
#  imposed_constraint(6201,"variant_set","mumps","openmp","False")
#  root("trilinos")
#  variant_condition(3362,"mumps","openmp")
#  variant_condition(5864,"trilinos","mumps")
#  variant_condition(5866,"trilinos","openmp")
#  variant_set("trilinos","mumps","True")
#  variant_set("trilinos","openmp","False")
#
# * Must build mumps with ~openmp set in the future 

# * Note, +zoltan +zoltan2 does not seem to add the package to the concetization list.
# [mkandes@login02 spack]$ spack spec -l trilinos@13.0.1 % gcc@10.2.0 +adios2 +amesos +amesos2 +anasazi +aztec ~basker +belos +boost ~chaco ~complex ~cuda ~cuda_rdc ~debug ~dtk +epetra +epetraext ~epetraextbtf ~epetraextexperimental ~epetraextgraphreorderings ~exodus +explicit_template_instantiation ~float +fortran +gtest +hdf5 ~hypre +ifpack +ifpack2 ~intrepid ~intrepid2 ~ipo ~isorropia +kokkos ~mesquite ~minitensor +ml +mpi +muelu ~mumps ~nox ~openmp ~phalanx ~piro +python ~rol ~rythmos +sacado ~scorec ~shards +shared ~shylu ~stk ~stokhos ~stratimikos ~strumpack +suite-sparse ~superlu +superlu-dist ~teko ~tempus +tpetra ~trilinoscouplings ~wrapper ~x11 +zoltan +zoltan2 "^adios2@2.7.1/$(spack find --format '{hash:7}' adios2@2.7.1 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi) ^hdf5@1.10.7/$(spack find --format '{hash:7}' hdf5@1.10.7 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^parmetis@4.0.3/$(spack find --format '{hash:7}' parmetis@4.0.3 % ${SPACK_COMPILER} ^openmpi@4.1.3) ^superlu-dist@7.1.1/$(spack find --format '{hash:7}' superlu-dist@7.1.1 % ${SPACK_COMPILER} ~int64 ^openmpi@4.1.3)"
#Input spec
#--------------------------------
#trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5~hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu~mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse~superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2
#    ^adios2@2.7.1%gcc@10.2.0+blosc+bzip2~dataman~dataspaces~endian_reverse+fortran+hdf5~ipo+mpi+pic+png+python+shared+ssc+sst+sz+zfp build_type=Release patches=8d301e8232baf4049b547f22bd73774309662017a62dac36360d2965907062bf arch=linux-rocky8-zen2
#        ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2
#        ^c-blosc@1.21.0%gcc@10.2.0+avx2~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#            ^lz4@1.9.3%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#            ^snappy@1.1.8%gcc@10.2.0~ipo+pic+shared build_type=RelWithDebInfo patches=c9cfecb1f7a623418590cf4e00ae7d308d1c3faeb15046c2e5090e38221da7cd arch=linux-rocky8-zen2
#            ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#            ^zstd@1.5.0%gcc@10.2.0~programs arch=linux-rocky8-zen2
#        ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2
#            ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#            ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#            ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
#                ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2
#                    ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                        ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                        ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                    ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#                ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2
#                    ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#                ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2
#                ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2
#                ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2
#                    ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#            ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^libfabric@1.13.2%gcc@10.2.0~debug~kdreg fabrics=sockets,tcp,udp arch=linux-rocky8-zen2
#        ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2
#        ^libpng@1.6.37%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^py-mpi4py@3.1.2%gcc@10.2.0 arch=linux-rocky8-zen2
#            ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2
#                ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2
#                    ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                        ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2
#                    ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2
#                ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^py-numpy@1.20.3%gcc@10.2.0+blas+lapack patches=873745d7b547857fcfec9cae90b09c133b42a4f0c23b6c2d84cf37e2dd816604 arch=linux-rocky8-zen2
#            ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#            ^py-setuptools@58.2.0%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^sz@2.1.12%gcc@10.2.0~fortran~hdf5~ipo~netcdf~pastri~python~random_access+shared~stats~time_compression build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^zfp@0.5.5%gcc@10.2.0~aligned~c~cuda~fasthash~fortran~ipo~openmp~profile~python+shared~strided~twoway bsws=64 build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#    ^boost@1.77.0%gcc@10.2.0+atomic+chrono~clanglibcpp~container~context~coroutine+date_time~debug+exception~fiber+filesystem+graph~icu+iostreams+locale+log+math~mpi+multithreaded+numpy+pic+program_options+python+random+regex+serialization+shared+signals~singlethreaded+system~taggedlayout+test+thread+timer~versionedlayout+wave cxxstd=98 patches=93f4aad8f88d1437e50d95a2d066390ef3753b99ef5de24f7a46bc083bd6df06,b8569d7d4c3ef0501a39857126a2b0a88519bf256c29f3252a6958916ce82255 visibility=hidden arch=linux-rocky8-zen2
#        ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2
#        ^py-numpy@1.20.3%gcc@10.2.0+blas+lapack patches=873745d7b547857fcfec9cae90b09c133b42a4f0c23b6c2d84cf37e2dd816604 arch=linux-rocky8-zen2
#            ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#            ^py-setuptools@58.2.0%gcc@10.2.0 arch=linux-rocky8-zen2
#                ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2
#                    ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2
#                        ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                            ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2
#                        ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2
#                            ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#                    ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2
#                        ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#                        ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#                            ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#                            ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#                        ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2
#                    ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2
#                    ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#                    ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2
#                    ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#        ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2
#    ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#    ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#        ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#        ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#        ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
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
#                ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#        ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#
#Concretized
#--------------------------------
#cj2zoib  trilinos@13.0.1%gcc@10.2.0+adios2+amesos+amesos2+anasazi+aztec~basker+belos+boost~chaco~complex~cuda~cuda_rdc~debug~dtk+epetra+epetraext~epetraextbtf~epetraextexperimental~epetraextgraphreorderings~exodus+explicit_template_instantiation~float+fortran+gtest+hdf5~hypre+ifpack+ifpack2~intrepid~intrepid2~ipo~isorropia+kokkos~mesquite~minitensor+ml+mpi+muelu~mumps~nox~openmp~phalanx~piro+python~rol~rythmos+sacado~scorec~shards+shared~shylu~stk~stokhos~stratimikos~strumpack+suite-sparse~superlu+superlu-dist~teko~tempus+tpetra~trilinoscouplings~wrapper~x11+zoltan+zoltan2 build_type=RelWithDebInfo cuda_arch=none cxxstd=14 gotype=long_long arch=linux-rocky8-zen2
#uoamkt4      ^adios2@2.7.1%gcc@10.2.0+blosc+bzip2~dataman~dataspaces~endian_reverse+fortran+hdf5~ipo+mpi+pic+png+python+shared+ssc+sst+sz+zfp build_type=Release patches=8d301e8232baf4049b547f22bd73774309662017a62dac36360d2965907062bf arch=linux-rocky8-zen2
#p7pyqr6          ^bzip2@1.0.8%gcc@10.2.0~debug~pic+shared arch=linux-rocky8-zen2
#ywzglk6          ^c-blosc@1.21.0%gcc@10.2.0+avx2~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#2uhqwjr              ^lz4@1.9.3%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#lyz7mj5              ^snappy@1.1.8%gcc@10.2.0~ipo+pic+shared build_type=RelWithDebInfo patches=c9cfecb1f7a623418590cf4e00ae7d308d1c3faeb15046c2e5090e38221da7cd arch=linux-rocky8-zen2
#mi7anth              ^zlib@1.2.11%gcc@10.2.0+optimize+pic+shared arch=linux-rocky8-zen2
#ofxb735              ^zstd@1.5.0%gcc@10.2.0~programs arch=linux-rocky8-zen2
#mw55rfb          ^hdf5@1.10.7%gcc@10.2.0+cxx+fortran+hl~ipo+java+mpi+shared+szip~threadsafe+tools api=default build_type=RelWithDebInfo arch=linux-rocky8-zen2
#o2k7zwt              ^libaec@1.0.5%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#fafnw7b              ^numactl@2.0.14%gcc@10.2.0 patches=4e1d78cbbb85de625bad28705e748856033eaafab92a66dffd383a3d7e00cc94,62fc8a8bf7665a60e8f4c93ebbd535647cebf74198f7afafec4c085a8825c006,ff37630df599cfabf0740518b91ec8daaf18e8f288b19adaae5364dc1f6b2296 arch=linux-rocky8-zen2
#ga6akx5              ^openjdk@11.0.12_7%gcc@10.2.0 arch=linux-rocky8-zen2
#vi547c4              ^openmpi@4.1.3%gcc@10.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~java+legacylaunchers+lustre~memchecker+pmi+pmix~singularity~sqlite3+static+thread_multiple+vt+wrapper-rpath fabrics=ucx schedulers=slurm arch=linux-rocky8-zen2
#7zpjmpk                  ^hwloc@2.6.0%gcc@10.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~opencl+pci~rocm+shared arch=linux-rocky8-zen2
#452ss2r                      ^libpciaccess@0.16%gcc@10.2.0 arch=linux-rocky8-zen2
#7y6keuv                      ^libxml2@2.9.12%gcc@10.2.0~python arch=linux-rocky8-zen2
#qslwinr                          ^libiconv@1.16%gcc@10.2.0 libs=shared,static arch=linux-rocky8-zen2
#fw6lna7                          ^xz@5.2.5%gcc@10.2.0~pic libs=shared,static arch=linux-rocky8-zen2
#vinqgvk                      ^ncurses@6.2%gcc@10.2.0~symlinks+termlib abi=none arch=linux-rocky8-zen2
#awzh7qi                  ^libevent@2.1.8%gcc@10.2.0+openssl arch=linux-rocky8-zen2
#pppnp3p                      ^openssl@1.1.1k%gcc@10.2.0~docs certs=system arch=linux-rocky8-zen2
#l2pamet                  ^lustre@2.12.8%gcc@10.2.0 arch=linux-rocky8-zen2
#zkreqqz                  ^openssh@8.7p1%gcc@10.2.0 arch=linux-rocky8-zen2
#5eimf7s                      ^libedit@3.1-20210216%gcc@10.2.0 arch=linux-rocky8-zen2
#fmhavjk                  ^pmix@3.2.1%gcc@10.2.0~docs+pmi_backwards_compatibility~restful arch=linux-rocky8-zen2
#rwpsuvv                  ^slurm@20.02.7%gcc@10.2.0~gtk~hdf5~hwloc~mariadb~pmix+readline~restd sysconfdir=PREFIX/etc arch=linux-rocky8-zen2
#pfpkgqi                  ^ucx@1.10.1%gcc@10.2.0~assertions~cm~cma~cuda~dc~debug~dm~gdrcopy~ib-hw-tm~java~knem~logging~mlx5-dv+optimizations~parameter_checking+pic~rc~rocm+thread_multiple~ud~xpmem cuda_arch=none arch=linux-rocky8-zen2
#req2hk3                      ^rdma-core@41.0%gcc@10.2.0~ipo build_type=RelWithDebInfo arch=linux-rocky8-zen2
#qoshsyz              ^pkgconf@1.8.0%gcc@10.2.0 arch=linux-rocky8-zen2
#rjpv734          ^libfabric@1.13.2%gcc@10.2.0~debug~kdreg fabrics=sockets,tcp,udp arch=linux-rocky8-zen2
#44wrz3l          ^libffi@3.3%gcc@10.2.0 patches=26f26c6f29a7ce9bf370ad3ab2610f99365b4bdd7b82e7c31df41a3370d685c0 arch=linux-rocky8-zen2
#34ctsc6          ^libpng@1.6.37%gcc@10.2.0 arch=linux-rocky8-zen2
#7tsq57n          ^py-mpi4py@3.1.2%gcc@10.2.0 arch=linux-rocky8-zen2
#dtdsuje              ^python@3.8.12%gcc@10.2.0+bz2+ctypes+dbm~debug+libxml2+lzma~nis+optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tix~tkinter~ucs4+uuid+zlib patches=0d98e93189bc278fbc37a50ed7f183bd8aaf249a8e1670a465f0db6bb4f8cf87,4c2457325f2b608b1b6a2c63087df8c26e07db3e3d493caf36a56f0ecf6fb768,f2fd060afc4b4618fe8104c4c5d771f36dc55b1db5a4623785a4ea707ec72fb4 arch=linux-rocky8-zen2
#qlg33ty                  ^expat@2.4.1%gcc@10.2.0+libbsd arch=linux-rocky8-zen2
#7e7gjsr                      ^libbsd@0.11.3%gcc@10.2.0 arch=linux-rocky8-zen2
#aa4a3m6                          ^libmd@1.0.3%gcc@10.2.0 arch=linux-rocky8-zen2
#ziinuas                  ^gdbm@1.19%gcc@10.2.0 arch=linux-rocky8-zen2
#hwm2pls                      ^readline@8.1%gcc@10.2.0 arch=linux-rocky8-zen2
#p66ksc2                  ^gettext@0.21%gcc@10.2.0+bzip2+curses+git~libunistring+libxml2+tar+xz arch=linux-rocky8-zen2
#evlo35q                      ^tar@1.34%gcc@10.2.0 arch=linux-rocky8-zen2
#ph73vfu                  ^sqlite@3.36.0%gcc@10.2.0+column_metadata+fts+functions+rtree arch=linux-rocky8-zen2
#pyzqbni                  ^util-linux-uuid@2.36.2%gcc@10.2.0 arch=linux-rocky8-zen2
#jzl32sl          ^py-numpy@1.20.3%gcc@10.2.0+blas+lapack patches=873745d7b547857fcfec9cae90b09c133b42a4f0c23b6c2d84cf37e2dd816604 arch=linux-rocky8-zen2
#fxzqxj3              ^openblas@0.3.17%gcc@10.2.0~bignuma~consistent_fpcsr~ilp64+locking+pic+shared threads=none arch=linux-rocky8-zen2
#4tbrdbw              ^py-setuptools@58.2.0%gcc@10.2.0 arch=linux-rocky8-zen2
#qkl3zt3          ^sz@2.1.12%gcc@10.2.0~fortran~hdf5~ipo~netcdf~pastri~python~random_access+shared~stats~time_compression build_type=RelWithDebInfo arch=linux-rocky8-zen2
#fqzonlg          ^zfp@0.5.5%gcc@10.2.0~aligned~c~cuda~fasthash~fortran~ipo~openmp~profile~python+shared~strided~twoway bsws=64 build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#nn45ql3      ^boost@1.77.0%gcc@10.2.0+atomic+chrono~clanglibcpp~container~context~coroutine+date_time~debug+exception~fiber+filesystem+graph~icu+iostreams+locale+log+math~mpi+multithreaded+numpy+pic+program_options+python+random+regex+serialization+shared+signals~singlethreaded+system~taggedlayout+test+thread+timer~versionedlayout+wave cxxstd=98 patches=93f4aad8f88d1437e50d95a2d066390ef3753b99ef5de24f7a46bc083bd6df06,b8569d7d4c3ef0501a39857126a2b0a88519bf256c29f3252a6958916ce82255 visibility=hidden arch=linux-rocky8-zen2
#ehft3km      ^cmake@3.21.4%gcc@10.2.0~doc+ncurses+openssl+ownlibs~qt build_type=Release arch=linux-rocky8-zen2
#k3rq6ok      ^metis@5.1.0%gcc@10.2.0~gdb~int64~real64+shared build_type=Release patches=4991da938c1d3a1d3dea78e49bbebecba00273f98df2a656e38b83d55b281da1,b1225da886605ea558db7ac08dd8054742ea5afe5ed61ad4d0fe7a495b1270d2 arch=linux-rocky8-zen2
#dt5p7vl      ^parmetis@4.0.3%gcc@10.2.0~gdb~int64~ipo+shared build_type=RelWithDebInfo patches=4f892531eb0a807eb1b82e683a416d3e35154a455274cf9b162fb02054d11a5b,50ed2081bc939269689789942067c58b3e522c269269a430d5d34c00edbc5870,704b84f7c7444d4372cb59cca6e1209df4ef3b033bc4ee3cf50f369bce972a9d arch=linux-rocky8-zen2
#wfqrwxl      ^suite-sparse@5.10.1%gcc@10.2.0~cuda~openmp+pic~tbb arch=linux-rocky8-zen2
#265iriy          ^gmp@6.2.1%gcc@10.2.0 arch=linux-rocky8-zen2
#dljfj3c              ^autoconf@2.69%gcc@10.2.0 patches=35c449281546376449766f92d49fc121ca50e330e60fefcfc9be2af3253082c2,7793209b33013dc0f81208718c68440c5aae80e7a1c4b8d336e382525af791a7,a49dd5bac3b62daa0ff688ab4d508d71dbd2f4f8d7e2a02321926346161bf3ee arch=linux-rocky8-zen2
#3a6q35d                  ^m4@1.4.19%gcc@10.2.0+sigsegv patches=9dc5fbd0d5cb1037ab1e6d0ecc74a30df218d0a94bdd5a02759a97f62daca573,bfdffa7c2eb01021d5849b36972c069693654ad826c1a20b53534009a4ec7a89 arch=linux-rocky8-zen2
#6yffi2u                      ^libsigsegv@2.13%gcc@10.2.0 arch=linux-rocky8-zen2
#gcv3szh                  ^perl@5.34.0%gcc@10.2.0+cpanm+shared+threads arch=linux-rocky8-zen2
#ehwhi27                      ^berkeley-db@18.1.40%gcc@10.2.0+cxx~docs+stl patches=b231fcc4d5cff05e5c3a4814f6a5af0e9a966428dc2176540d2c05aff41de522 arch=linux-rocky8-zen2
#7rq2ldo              ^automake@1.16.3%gcc@10.2.0 arch=linux-rocky8-zen2
#6h2nwcr              ^libtool@2.4.6%gcc@10.2.0 arch=linux-rocky8-zen2
#sumogwv          ^mpfr@4.1.0%gcc@10.2.0 arch=linux-rocky8-zen2
#jlg63di              ^autoconf-archive@2019.01.06%gcc@10.2.0 arch=linux-rocky8-zen2
#4hsrxvs              ^texinfo@6.5%gcc@10.2.0 patches=12f6edb0c6b270b8c8dba2ce17998c580db01182d871ee32b7b6e4129bd1d23a,1732115f651cff98989cb0215d8f64da5e0f7911ebf0c13b064920f088f2ffe1 arch=linux-rocky8-zen2
#jqsri3x      ^superlu-dist@7.1.1%gcc@10.2.0~cuda~int64~ipo~openmp+shared build_type=RelWithDebInfo cuda_arch=none arch=linux-rocky8-zen2
#nvkzwlg      ^swig@4.0.2%gcc@10.2.0 arch=linux-rocky8-zen2
#kewkz3p          ^pcre@8.44%gcc@10.2.0~jit+multibyte+utf arch=linux-rocky8-zen2

# when forcing zoltan as a dependency ...
# Concretized
#--------------------------------
#==> Error: Package trilinos does not depend on zoltan@3.83%gcc@10.2.0~debug+fortran~int64+mpi+parmetis+shared arch=linux-rocky8-zen2
#
# * Check with spack devs eventually, maybe a pull request here.

# Build likely failing due to use of ~int64 in certain packages ...
#              st.cpp: In member function 'int Amesos_Superludist::Factor()':
#  >> 2993    /tmp/mkandes/spack-stage/spack-stage-trilinos-13.0.1-n5xwn72sviuei
#             s4763daeewivkmraktt/spack-src/packages/amesos/src/Amesos_Superludi
#             st.cpp:459:20: error: cannot convert '__gnu_cxx::__alloc_traits<st
#             d::allocator<int>, int>::value_type*' {aka 'int*'} to 'int_t*' {ak
#             a 'long int*'}
#     2994      459 |         &Aval_[0], &Ai_[0], &Ap_[0],
#     2995    In file included from /tmp/mkandes/spack-stage/spack-stage-trilino
#             s-13.0.1-n5xwn72sviueis4763daeewivkmraktt/spack-src/packages/ameso
#             s/src/Amesos_Superludist.cpp:38:
#     2996    /home/mkandes/cm/shared/apps/spack/0.17.3/cpu/opt/spack/linux-rock
#             y8-zen2/gcc-10.2.0/superlu-dist-7.1.1-jqsri3xznegbq32rv4vnuloqdsci
#             b3np/include/superlu_ddefs.h:354:28: note:   initializing argument
#              8 of 'void dCreate_CompRowLoc_Matrix_dist(SuperMatrix*, int_t, in
#             t_t, int_t, int_t, int_t, double*, int_t*, int_t*, Stype_t, Dtype_
#             t, Mtype_t)'
#     2997      354 |           int_t, double *, int_t *, int_t *,
#     2998          |                            ^~~~~~~
#  >> 2999    /tmp/mkandes/spack-stage/spack-stage-trilinos-13.0.1-n5xwn72sviuei
#             s4763daeewivkmraktt/spack-src/packages/amesos/src/Amesos_Superludi
#             st.cpp:491:54: error: cannot convert 'int*' to 'int_t*' {aka 'long
#              int*'} in assignment
#     3000      491 |       PrivateSuperluData_->ScalePermstruct_.perm_c = perm_
#             c_;
#     3001          |                                                      ^~~~~
#             ~~
#     3002          |                                                      |
#     3003          |                                                      int*
#  >> 3004    /tmp/mkandes/spack-stage/spack-stage-trilinos-13.0.1-n5xwn72sviuei
#             s4763daeewivkmraktt/spack-src/packages/amesos/src/Amesos_Superludi
#             st.cpp:502:54: error: cannot convert 'int*' to 'int_t*' {aka 'long
#              int*'} in assignment
#     3005      502 |       PrivateSuperluData_->ScalePermstruct_.perm_r = perm_
#             r_;
#     3006

declare -xr SPACK_PACKAGE='trilinos@13.0.1'
declare -xr SPACK_COMPILER='gcc@10.2.0'
declare -xr SPACK_VARIANTS='+adios2 +amesos +amesos2 +anasazi +aztec ~basker +belos +boost ~chaco ~complex ~cuda ~cuda_rdc ~debug ~dtk +epetra +epetraext ~epetraextbtf ~epetraextexperimental ~epetraextgraphreorderings ~exodus +explicit_template_instantiation ~float +fortran +gtest +hdf5 ~hypre +ifpack +ifpack2 ~intrepid ~intrepid2 ~ipo ~isorropia +kokkos ~mesquite ~minitensor +ml +mpi +muelu ~mumps ~nox ~openmp ~phalanx ~piro +python ~rol ~rythmos +sacado ~scorec ~shards +shared ~shylu ~stk ~stokhos ~stratimikos ~strumpack +suite-sparse ~superlu +superlu-dist ~teko ~tempus +tpetra ~trilinoscouplings ~wrapper ~x11 ~zoltan ~zoltan2'
declare -xr SPACK_DEPENDENCIES="^adios2@2.7.1/$(spack find --format '{hash:7}' adios2@2.7.1 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^boost@1.77.0/$(spack find --format '{hash:7}' boost@1.77.0 % ${SPACK_COMPILER} ~mpi) ^hdf5@1.10.7/$(spack find --format '{hash:7}' hdf5@1.10.7 % ${SPACK_COMPILER} +mpi ^openmpi@4.1.3) ^parmetis@4.0.3/$(spack find --format '{hash:7}' parmetis@4.0.3 % ${SPACK_COMPILER} ^openmpi@4.1.3) ^superlu-dist@7.1.1/$(spack find --format '{hash:7}' superlu-dist@7.1.1 % ${SPACK_COMPILER} ~int64 ^openmpi@4.1.3)"
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

#sbatch --dependency="afterok:${SLURM_JOB_ID}" ''

sleep 20
