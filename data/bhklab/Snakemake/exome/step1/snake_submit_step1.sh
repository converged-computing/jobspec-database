#!/bin/bash

#SBATCH --job-name=snake_step1
#SBATCH --mem=3000M
#SBATCH -c 1
#SBATCH -t 72:00:00

source /cluster/home/amammoli/.bashrc
module load python3

output_dir="/cluster/home/amammoli/exome" snakemake -s /path/to/gitclone/step1/Snakefile_step1 --latency-wait 100 -j 100 --cluster 'sbatch -t {params.runtime} --mem={resources.mem_mb} -c {threads}'
