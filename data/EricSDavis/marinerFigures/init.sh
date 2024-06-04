#! /bin/bash
#SBATCH -J marinerFigures
#SBATCH -t 10-00:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1  
#SBATCH -p general
#SBATCH --mem=2gb
#SBATCH -o "%x-%j.out"

## Exit if any command fails
set -e

## Make directory for slurm logs
mkdir -p logs

## Load required modules
module load python/3.7.14

## Create and activate virtual env to install snakemake
python3 -m venv env &&\
  source env/bin/activate &&\
  pip3 install snakemake

## Execute workflow
snakemake \
  --cluster "sbatch -J {rule} \
                    --mem={resources.mem} \
                    -t {resources.runtime} \
                    -o logs/{rule}_%j.out \
                    -e logs/{rule}_%j.out" \
  -j 100 \
  --rerun-incomplete