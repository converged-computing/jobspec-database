#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=log_alignment_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=example@gmail.com
#SBATCH --job-name=example_name

## SET EMAIL ADDRESS & JOB NAME ABOVE

##  move this file to WGS folder and run with `sbatch` command

module load nextflow

nextflow run /gpfs/data/davolilab/data/WGS/01-scripts/wgs-copynumber-workflow/wgs-alignment.nf -with-report report-nextflow-alignment-$(date +"%Y%m%d%H%m").html -resume