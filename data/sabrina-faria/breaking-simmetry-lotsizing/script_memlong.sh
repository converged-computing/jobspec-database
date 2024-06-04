#PBS -N main
#PBS -q memlong
#PBS -l nodes=1:ppn=128
#PBS -e outputs/erros_memlong
#PBS -o outputs/saidas_memlong
#PBS -m abe
#PBS -M s209376@dac.unicamp.br

module load python/3.8.11-intel-2021.3.0
module load openmpi/4.1.1-gcc-9.4.0
source py38/bin/activate

cd $PBS_O_WORKDIR
mpirun -np 128 python -m mpi4py.futures main.py