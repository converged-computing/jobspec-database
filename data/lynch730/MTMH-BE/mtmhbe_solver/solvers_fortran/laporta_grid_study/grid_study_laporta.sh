#!/bin/sh
# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=01:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=18   # 36 processor core(s) per node
#SBATCH --mem=20G   # maximum memory per node
#SBATCH --job-name="for_gst_2log"
#SBATCH --output="grid_study_laporta_ism2_fortran.out"

MATLAB_VER=R2021a
cd /home/lynch/boltzmann_solvers/mtmhbe_solver/solvers_fortran/laporta_grid_study
module load intel
module load intel-mkl
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/shared/hpc/matlab/$MATLAB_VER/bin/glnxa64:/shared/hpc/matlab/$MATLAB_VER/sys/os/glnxa64
export PATH=$PATH:/shared/hpc/matlab/$MATLAB_VER/extern/include

./grid_study_laporta_exe > /home/lynch/boltzmann_solvers/mtmhbe_solver/performance/laporta/grid_study/grid_sweep_data/laporta_mtmhbe_ET2_log_fortran.txt
