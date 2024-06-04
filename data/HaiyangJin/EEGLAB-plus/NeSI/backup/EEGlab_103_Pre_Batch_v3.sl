#!/bin/bash
#SBATCH --job-name=103      # job name (shows up in the queue)
#SBATCH --account=uoa00424     # Project Account
#SBATCH --time=06:00:00         # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=4096      # memory/cpu (in MB)
#SBATCH --ntasks=1              # number of tasks (e.g. MPI)
#SBATCH --cpus-per-task=2       # number of cores per task (e.g. OpenMP)
#SBATCH --partition=bigmem        # specify a partition
#SBATCH --hint=nomultithread    # don't use hyperthreading
#SBATCH --mail-type=END
#SBATCH --mail-user=hjin317@aucklanduni.ac.nz
#SBATCH --array=34    # Array definition 11,12,13,14 (step 1 and experiment number)
module load MATLAB/2017b
srun matlab -nodesktop -nosplash -r EEGlab_101_Pre_Batch_v3 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID