#PBS -S /bin/bash
#PBS -N pk_run
#PBS -V 
#PBS -l nodes=1:ppn=1
#PBS -l walltime=0:120:00
#PBS -l mem=5MB
#PBS -p 1023   


DIR="/global/homes/m/mjwilson/Spectre-MC/"
cd $DIR

export OMP_NUM_THREADS=16                      # Threads = processors.
export BRANCH=$(git symbolic-ref --short HEAD) # current Git branch

export GSL_RNG_TYPE="taus"
export GSL_RNG_SEED=123


module load gcc
module load gsl

export OMP_NUM_THREADS=32

# -pedantic -std=gnu11 -Wall -Wextra
gcc -o pk.o driver_pk.c -I/opt/cray/pe/fftw/3.3.8.2/haswell/include -L/opt/cray/pe/fftw/3.3.8.2/haswell/lib -fopenmp -lfftw3_omp -lfftw3 -lm -I/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq/include -L/global/common/sw/cray/cnl7/haswell/gsl/2.5/intel/19.0.3.199/7twqxxq/lib -lgsl -lgslcblas 

./pk.o ##  > output.txt
