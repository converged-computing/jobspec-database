#PBS -q GPUQ
#PBS -l nodes=srvslsgpu003:ppn=4:seriesGPUk
#PBS -N compile-python
#PBS -o Q.compile-python.log
#PBS -e Q.compile-python.err
#PBS -V

source $HOME/scripts/base.sh

module load compilers/gcc492
module load mpi/openmpi-1.8.8
module load cuda/Cuda-6.5

cd ${SOURCE_PREFIX}/Python-2.7.10
#./configure --prefix=$HOME/usr --enable-shared
./configure --prefix=${INSTALL_PREFIX}
make uninstall
make clean
make
make install
