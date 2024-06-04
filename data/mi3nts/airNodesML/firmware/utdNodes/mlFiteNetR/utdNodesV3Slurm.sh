#!/bin/bash -l

#SBATCH -J utdNodesCalV2  # Job name
#SBATCH -o logs/utdNodesCalV2.%j.out # Name of stdout output file (%j expands to jobId)
#SBATCH -e logs/utdNodesCalV2.%j.err # Error File Name 
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=lhw150030@utdallas.edu     # Where to send mail	
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 16                 # Total number of mpi tasks requested
#SBATCH --array=1-31          # Array ranks to run
#SBATCH -t 48:00:00           # Run time (hh:mm:ss) - 48 hours


echo Running calibration scripts for UTD Node: "$SLURM_ARRAY_TASK_ID"
echo Running on host: `hostname`
matlab -nodesktop -nodisplay -nosplash -r "utdNodesV3("$SLURM_ARRAY_TASK_ID")"

