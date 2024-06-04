#!/bin/bash
#SBATCH -J job               # sensible name for the job
#SBATCH --qos=debug                  # 4. Request a QoS
#SBATCH --partition=amd               # 3. Request a partition
#SBATCH -N 1                    # Allocate 2 nodes for the MPI job
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH -c 20
#SBATCH -t 00:10:00             # Upper time limit for the job
#SBATCH --time=0-00:30:00             # 7. Job execution duration limit day-hour:min:sec
#SBATCH --output=job/%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --error=job/%x_%j.err

module load intel
module load R
export LANG=C

R --vanilla -f testdopar.R
