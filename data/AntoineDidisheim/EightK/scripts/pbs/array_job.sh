#!/bin/bash
#PBS -P nz97          # The project to charge the job's resource usage to
#PBS -q gpuvolta            # Queue name
#PBS -l walltime=01:00:00   # Walltime
#PBS -l storage=scratch/my_project+gdata/xy11 # Required filesystems
#PBS -l mem=20GB            # Memory limit
#PBS -l ncpus=4             # Number of CPU cores
#PBS -l ngpus=2             # Number of GPUs
#PBS -l jobfs=500MB         # Local disk limit
#PBS -l software=matlab_anu # Required software licenses
#PBS -l wd                  # Start job in submission directory
#PBS -M your.email@example.com # Email notifications
#PBS -m abe                 # When to send the email
#PBS -N MyJobArray          # Job name
#PBS -o output/             # Standard output log path
#PBS -e error/              # Standard error log path
#PBS -J 1-10                # Creates an array job with indices from 1 to 10
#PBS -j oe                  # Merge standard output and error streams
#PBS -W depend=beforeok:123456:123457 # Job dependencies
#PBS -a 202309220900       # Delayed execution time

# Your computational commands here
module load python3/3.10.4
source ~/scratch/nz97/ad4734/venv/bin/activate
python my_script.py ${PBS_ARRAYID}
