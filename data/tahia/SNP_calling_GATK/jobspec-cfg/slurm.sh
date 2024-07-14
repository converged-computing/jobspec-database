#!/bin/bash
#SBATCH -J Job
#SBATCH -o Job.o%j
#SBATCH -e Job.e%j
#SBATCH -N 6
#SBATCH -n 11
#SBATCH --ntasks-per-node=2 
#SBATCH -p normal
#SBATCH -t 30:00:00
#SBATCH -A P.hallii_expression


module load launcher
ml intel/17.0.4
ml fastx_toolkit
ml bwa
ml picard
ml samtools
ml gatk/3.8.0
ml sratoolkit

CMD=$1

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=$CMD

$LAUNCHER_DIR/paramrun

echo "DONE";
date;

