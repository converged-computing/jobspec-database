#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
 
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name MC_Pi
#SBATCH --constraint=amr
 
########## Command Lines to Run ##########

# Commenting the following lines, because I compiled my code with default modules.
# module purge
# module load gcc/7.3.0-2.30 openmpi hdf5 python git
  
cd $SLURM_SUBMIT_DIR                   ### change to the directory where your code is located

# load modules

module purge
module load gcc/7.3.0-2.30 openmpi hdf5 python git

# Establish run name
RUN_NAME=$1

# compile mpi file

mpicxx -o $RUN_NAME src/Ser_PI_Calc_MPI.cpp

# run program

# Lower number of darts using all range of prcesses
DartValues=(1000 10000 100000 1000000 10000000)
RoundValues=(128)
TaskValues=(1 2 4 8 16 32 64)
# Loop over DartsValues and RoundsValues arrays
for Darts in "${DartValues[@]}"; do
    for Rounds in "${RoundValues[@]}"; do
        for Tasks in "${TaskValues[@]}"; do
            mpiexec -n "$Tasks" $RUN_NAME "$Darts" "$Rounds" "$RUN_NAME.csv"
        done
    done
done

# Higher number of darts only with more than 16 processes.
DartValues=(100000000 1000000000)
RoundValues=(128)
TaskValues=(16 32 64)
# Loop over DartsValues and RoundsValues arrays
for Darts in "${DartValues[@]}"; do
    for Rounds in "${RoundValues[@]}"; do
        for Tasks in "${TaskValues[@]}"; do
            mpiexec -n "$Tasks" $RUN_NAME "$Darts" "$Rounds" "$RUN_NAME.csv"
        done
    done
done

rm $RUN_NAME
scontrol show job $SLURM_JOB_ID     ### write job information to output file
