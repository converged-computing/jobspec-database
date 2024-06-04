#!/bin/sh
# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node
#SBATCH --mem=64G   # maximum memory per node
#SBATCH --job-name="fort_Ne_sweeo"
#SBATCH --output="fort_Ne_sweep.out"

MATLAB_VER=R2022a
cd /home/lynch/boltzmann_solvers/mtmhbe_solver/solvers_fortran/time-dependent-pulse-Neps
module load intel
module load intel-mkl
#export OMP_NUM_THREADS=18
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/shared/hpc/matlab/$MATLAB_VER/bin/glnxa64:/shared/hpc/matlab/$MATLAB_VER/sys/os/glnxa64
export PATH=$PATH:/shared/hpc/matlab/$MATLAB_VER/extern/include

./loki_b_pulse_exe > fort_Ne_sweep.txt
