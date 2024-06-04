#!/bin/bash -l

# Specific course queue and max wallclock time
#SBATCH -p course -t 2

# Defaults on Barkla (but set to be safe)
## Specify the current working directory as the location for executables/files
#SBATCH -D ./
## Export the current environment to the compute node
#SBATCH --export=ALL

# load modules intel compiler
module load compilers/intel/2019u5 

## SLURM terms
## nodes            relates to number of nodes
## ntasks-per-node  relates to MPI processes per node
## cpus-per-task    relates to OpenMP threads (per MPI process)

echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of threads or processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"

# parallel using OpenMP
# SRC = $1 is name of the source code as an arguemnt
SRC=$1

#sets the exe name as the sourcecode, and %% removes the ".c"
EXE=${SRC%%.c}.exe 

#deletes the existing executable (if it exists)
rm -f ${EXE} 

echo compiling $SRC to $EXE 

#compilation using intel compiler of sourcecode to exectuable.
icc -qopenmp -O0 -std=c99 $SRC  -o $EXE  
echo
echo ------------------------------------

#this tests if the file $EXE is present and executable
if test -x $EXE; then 
      # set number of threads
      
      # if '-c' not used then default to 1. SLURM_CPUS_PER_TASK is given by -c
      export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} 
      echo using ${OMP_NUM_THREADS} OpenMP threads
      echo
      echo 
      echo Multiple execution..
      echo
      echo

      # run multiple times. Because we have exported how many threads we're using, we just execute the file.
      for i in {1..5}; do ./${EXE}; done     
else
     echo $SRC did not built to $EXE
fi

