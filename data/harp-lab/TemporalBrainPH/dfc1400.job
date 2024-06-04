#!/bin/bash
#
#SBATCH --job-name=dfc1400
#SBATCH --output=dfc1400_job.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=amd-hdr100
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=4069

# load your Anaconda module here and activate your virtual environment (if needed)
set -e
source /home/ashovon/newaumri/matfiles/venv/bin/activate


# execute your python scripts (change it to whatever it needs to be for you).
python -u /home/ashovon/newaumri/matfiles/TemporalBrainPH/distance_calculation.py --data 1400 --method ws --start 1 --end 316 --distance y --mds y
