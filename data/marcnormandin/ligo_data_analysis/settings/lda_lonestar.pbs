#!/bin/bash
#SBATCH -J myMPI            # job name
#SBATCH -o myMPI.o%j        # output and error file name (%j expands to jobID)
#SBATCH -N 11                # number of nodes requested
#SBATCH -n 11               # total number of mpi tasks requested
#SBATCH -p development      # queue (partition) -- normal, development, etc.
#SBATCH -t 01:30:00         # run time (hh:mm:ss) - 1.5 hours

# Slurm email notifications are now working on Lonestar 5 
#SBATCH --mail-user=normandin.utb@gmail.com
#SBATCH --mail-type=begin   # email me when the job starts
#SBATCH --mail-type=end     # email me when the job finishes

# run the executable named a.out
export OMP_NUM_THREADS=24
cd $WORK
ibrun tacc_affinity ./lda_matlab_pso settings.cfg data_snr9_0232.map pso.cfg 66 data_snr9_0232.pso

