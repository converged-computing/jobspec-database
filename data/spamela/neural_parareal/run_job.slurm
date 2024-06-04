#!/bin/bash --login

# --- PBS job options (name, compute nodes, job time)
#PBS -N JOREK

# --- select_max=167 
#PBS -l select=1:ncpus=1:mpiprocs=48

# --- memory per node mem_max=123GB
# #PBS -l mem=64

# --- Walltime max 24h
#PBS -l walltime=10:59:00
# #PBS -l walltime=00:59:00

# --- Replace [budget code] below with your project code (e.g. t01)
#PBS -A FUA37_UKAEA_ML

# --- Queue
# #SBATCH --partition=skl_fua_dbg
#SBATCH --partition=skl_fua_prod
# #SBATCH --qos=skl_qos_fuabprod

### Set environment
#module load intel intelmpi mkl fftw szip
module purge
module load profile/archive
module load gnuplot
module load intel/pe-xe-2018--binary intelmpi/2018--binary \
        mkl/2018--binary \
        zlib/1.2.8--gnu--6.1.0 \
        szip/2.1--gnu--6.1.0 \
        fftw \
        hdf5/1.8.18--intelmpi--2018--binary \
        lapack/3.8.0--intel--pe-xe-2018--binary \
        blas/3.8.0--intel--pe-xe-2018--binary
module load python/3.9.4 
source /marconi_work/FUA37_UKAEA_ML/spamela/Parareal/jorek_parareal/venv/bin/activate
export OMP_NUM_THREADS=1
export I_MPI_PIN_MODE=lib
export OMP_STACKSIZE=512m


# --- Example command to run with a "fake" coarse solver, ie. JOREK itself but with a coarser grid
python3 ./run_parareal_jorek.py -np 40 -coarse_not_slurm -no_ref -chkpt "initial_run" -ip 0 -ic 0 > output.txt

# --- Example command to run with a "real" coarse solver, ie. PDEarena surrogate
python3 ./run_parareal_jorek.py -np 4  -ninn 5 -nonn 10 -coarse_not_slurm -no_ref -chkpt "initial_run" -multi_chkpt -coarse_not_jorek -ip 0 -ic 0 > output.txt
