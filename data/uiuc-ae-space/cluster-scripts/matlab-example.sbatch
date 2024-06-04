#!/usr/bin/bash
#####################################################################################################################
##                                                                                                                 ##
##                                                    Campus Cluster                                               ##
##                                              MATLAB example batch script                                        ##
##                                                                                                                 ##
## SLURM Options (To view, run the following command below)                                                        ##
##     man sbatch                                                                                                  ##
##                                                                                                                 ##
#####################################################################################################################

#SBATCH --time=0-04:00:00                                # Job run time (d-hh:mm:ss)
#SBATCH --nodes=1                                        # Number of nodes
#SBATCH --ntasks-per-node=40                             # Number of task (cores/ppn) per node (16/20/24/28/40)
#SBATCH --job-name=matlab-example                        # Name of batch job
#SBATCH --partition=eng-research                         # Partition (queue)           
#SBATCH --output=logs/%x_%j.out                          # Name of batch job output file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err                           # Name of batch job error file (%x = job name, %j = job ID)
#SBATCH --mail-user=netID@illinois.edu                   # Send email notifications
#SBATCH --mail-type=BEGIN,FAIL,END                       # Type of email notifications to send

#####################################################################################################################

# Change to the scratch storage to prepare for job start (root directory where files exist in reality)
cd /scratch/users/netID/matlab-file-directory/

# Clear the value set in the DISPLAY environment variable to run the CLI version of MATLAB
unset DISPLAY
# Load MATLAB module (Enable MATLAB in user environment)
module load matlab/9.7

# Run MATLAB code and redirect output into a file whose name includes the Jobname and JobID
matlab -nodisplay -r main.m >& logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.oe
