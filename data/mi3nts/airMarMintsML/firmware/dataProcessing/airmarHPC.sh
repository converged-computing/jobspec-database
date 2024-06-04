#SBATCH -J airMar        # Job name
#SBATCH -o airMar.%j.out # Name of stdout output file (%j expands to jobId)
#SBATCH -e airMar.%j.err # Error File Name 
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 16                 # Total number of mpi tasks requested
#SBATCH --array=1-1          # Array ranks to run
#SBATCH -t 48:00:00           # Run time (hh:mm:ss) - 24 hours

ml load matlab
echo Running calibration scripts for UTD Node: "$SLURM_ARRAY_TASK_ID"
echo Running on host: `hostname`
matlab -nodesktop -nodisplay -nosplash -r "try airmar0001; catch; end; quit"