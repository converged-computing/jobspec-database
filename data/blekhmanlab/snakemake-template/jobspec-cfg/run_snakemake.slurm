#!/bin/bash -e
#SBATCH -t 45:00:00 -N 1
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH -p blekhman
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=email@EMAIL_GOES_HERE.com

# load python3
module load python

# create a virtual environment if one doesn't exist
if [ ! -d "./venv" ]; then
    python -m venv venv
fi

# activate the virtual environment
source venv/bin/activate

# if snakemake isn't installed, install it
if [ ! -f "./venv/bin/snakemake" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# run snakemake
snakemake --jobs 25 --slurm --default-resources slurm_account=blekhman slurm_partition=blekhman

# to submit this job:
# sbatch --job-name=next -o nextone.log run_snakemake.slurm
# this will create a log called "nextone.log" in the current directory
