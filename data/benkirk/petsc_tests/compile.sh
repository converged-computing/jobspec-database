#!/usr/bin/env bash
#PBS -q main
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=60:mpiprocs=1:ompthreads=60:ngpus=2

# Handle arguments
user_args=( "$@" )

ncar_stack=no
from_git=no
from_tarball=yes

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --ncar-stack)
            ncar_stack=yes
            ;;
        --from-git)
            from_git=yes
            from_tarball=no
            ;;
        --from-tarball)
            from_git=no
            from_tarball=yes
            ;;
        *)
            ;;
    esac

    shift
done

cat >config_env.sh <<EOF
top_dir=$(pwd)
EOF

# build using ncarenv
if [[ ${ncar_stack} == "yes" ]]; then
    cat >>config_env.sh <<EOF
module reset
module load gcc/11.2.0 cuda cray-libsci
module list
for tool in CC cc ftn gcc mpiexec; do
    which \${tool}
done
export BUILD_CLASS="ncarenv"
BLAS_LAPACK="-L${CRAY_LIBSCI_DIR}/cray/9.0/x86_64/lib -lsci_cray"
EOF

# build using crayenv
else
    cat >>config_env.sh <<EOF
module purge
module load crayenv
module load PrgEnv-gnu/8.3.2 craype-x86-rome craype-accel-nvidia80 libfabric cray-pals cpe-cuda cray-libsci
module list
for tool in CC cc ftn gcc mpiexec; do
    which \${tool}
done
export BUILD_CLASS="crayenv"
BLAS_LAPACK="-L${CRAY_LIBSCI_DIR}/cray/9.0/x86_64/lib -lsci_cray"
EOF
fi

cat >>config_env.sh  <<EOF
package="petsc"
version="3.17.4"
src_dir_tar=\${package}-\${version}
src_dir_git=\${package}.git
tarball=\${package}-\${version}.tar.gz
inst_dir=$(pwd)/install-\${BUILD_CLASS}
export PETSC_ARCH=\${BUILD_CLASS}

# Enable GPU support in the MPI library
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
EOF

cat config_env.sh && . config_env.sh || exit 1

if [[ ${from_tarball} == "yes" ]]; then
    src_dir=${src_dir_tar}
    [ -f ${tarball} ] || wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/${tarball} || exit 1
    [ -d ${src_dir} ] || tar zxf ${tarball} || exit 1
else
    src_dir=${src_dir_git}
    [ -d ${src_dir} ] || git clone -b release https://gitlab.com/petsc/petsc.git ${src_dir}
fi

export PETSC_DIR=${top_dir}/${src_dir}

cat >>config_env.sh <<EOF
export PETSC_DIR=${PETSC_DIR}
EOF

env | sort | uniq | egrep -v "_LM|_ModuleTable"

cd ${PETSC_DIR} && pwd || exit 1


mkdir -p ${top_dir}/downloads
rm -rf ./${PETSC_ARCH}
./configure \
    --with-cc=$(which cc) --COPTFLAGS="-O3" \
    --with-cxx=$(which CC) --CXXOPTFLAGS="-O3" \
    --with-fc=$(which ftn) --FOPTFLAGS="-O3" \
    --with-cmake-dir=/glade/u/apps/common/22.08/spack/opt/spack/cmake/3.23.2/gcc/7.5.0 \
    --with-cmake-exec=/glade/u/apps/common/22.08/spack/opt/spack/cmake/3.23.2/gcc/7.5.0/bin/cmake \
    --enable-cuda --CUDAOPTFLAGS="-O3" \
    --with-packages-download-dir="${top_dir}/downloads" \
    --with-shared-libraries --with-debugging=0 \
    --with-blaslapack-lib="${BLAS_LAPACK}" \
    --with-hypre=1        --download-hypre=yes \
    --with-metis=1        --download-metis=yes \
    --with-ml=1           --download-ml=yes \
    --with-parmetis=1     --download-parmetis=yes \
    --with-scalapack=1    --download-scalapack=yes \
    --with-sowing=0 \
    --with-spooles=1      --download-spooles=yes \
    --with-suitesparse=1  --download-suitesparse=yes \
    --with-superlu=1      --download-superlu=yes \
    --with-superlu_dist=1 --download-superlu_dist="${top_dir}/downloads/b430c074a19bdfd897d5e2a285a85bc819db12e5.tar.gz" \
    --with-triangle=1     --download-triangle=yes \
    --with-tetgen=1       --download-tetgen=yes \
    --with-viennacl=1     --download-viennacl=yes \
        || exit 1


make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all || exit 1
#rm -rf ${inst_dir} && make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} install || exit 1
make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} check || exit 1

cd ${PETSC_DIR}/src/snes/tutorials/ || exit 1
gmake V=1 ex12 ex19

echo && echo && echo "Done at $(date)"

# mpirun -np 4 ./ex19 -da_refine 3 -snes_monitor -dm_mat_type mpiaijcusparse -dm_vec_type mpicuda -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -ksp_monitor  -mg_levels_ksp_max_it 3 -log_view
