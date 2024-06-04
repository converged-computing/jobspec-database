#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=CAN_EXHAUSTIVE

### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

### REQUIRED. Specify the PI group for this job
#SBATCH --account=sprinkjm

### Optional. Request email when job begins and ends

### Specify high priority jobs
#SBATCH --qos=user_qos_sprinkjm

# SBATCH --mail-type=ALL

### Optional. Specify email address to use for notification
# SBATCH --mail-user=rahulbhadani@email.arizona.edu

### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard

### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=70

### REQUIRED. Set the memory required for this job.
#SBATCH --mem=300gb

### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=200:00:00


# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------

pwd; hostname; date
now=$(date +"%Y_%m_%d_%H_%M_%S")
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
module load matlab/r2020b
ulimit -u 63536
echo $now
matlab -nodisplay -nosplash -softwareopengl < /home/u27/rahulbhadani/CANExhaustiveSearch/CorrSignal_impl.m > /home/u27/rahulbhadani/CANExhaustiveSearch/output_pbs_$now.txt
date
