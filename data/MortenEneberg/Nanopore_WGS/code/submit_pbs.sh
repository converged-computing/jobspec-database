#!/bin/sh
#PBS -N nanopore_wgs ## Name of the job for the scheduler
#PBS -W group_list=cu_00014 -A cu_00014 ## name of the allocation (who is paying for the compute time)

### Number of nodes
#PBS -l nodes=1:fatnode:ppn=40
### Memory
#PBS -l mem=700gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 12 hours)
#PBS -l walltime=170:00:00

#PBS -M menie@bio.aau.dk  ## send email notifications to umich email listed
#PBS -m abe                ## when to send email a=abort b=job begin e=job end

### Here follows the user commands:
# Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes

# Load all required modules for the job
module load tools
module load snakemake/7.18.2
module load mamba-org/mamba/0.24.0 

WD="/home/projects/cu_00014/data/sepseq_WGS/analysis/all_samples/Nanopore_WGS/"
cd $WD

#  Put your job commands after this line. Load all required modules before submitting this script.
snakemake --latency-wait 90 -s Snakefile --configfile config/config_pbs.yaml --cores 40 --use-conda --conda-frontend mamba --keep-going