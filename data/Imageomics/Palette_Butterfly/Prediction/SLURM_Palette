#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --job-name=segment_test
#SBATCH --time=00:20:00
#SBATCH --ntasks=4
#
# Runs Snakemake pipeline that detect the color palette in the case of buttefly
# There are three required positional arguments.
# Usage:
# sbatch SLURM_Snake <SNAKEFILE> <WORKDIR> <INPUTCSV>
# - SNAKEFILE - snakefile you want to used
# - WORKDIR - Snakemake working directory - contains output files
# - INPUTCSV - Path to input CSV file specifying images to process

# Stop if a command fails (non-zero exit status)
set -e

# Verify command line arguments
WORKDIR=$1


# Activate Snakemake environment
module load miniconda3/4.10.3-py37
# Activate using source per OSC instructions
source activate snakemake

# Run pipeline using Snakemake
snakemake \
    --cores $SLURM_NTASKS \
    --use-singularity \
    --directory $WORKDIR \

chmod -R 774 $WORKDIR
