#!/bin/sh
#PBS -N nanopore_wgs ## Name of the job for the scheduler
#PBS -W group_list=cu_00014 -A cu_00014 ## name of the allocation (who is paying for the compute time)

### Number of nodes
#PBS -l nodes=1:fatnode:ppn=40
### Memory
#PBS -l mem=700gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 12 hours)
#PBS -l walltime=20:00:00

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
module load gtdbtk/2.3.2

WD="/home/projects/cu_00014/data/sepseq_WGS/analysis/Nanopore_WGS/"
cd $WD

sample_dir="data/20_flye_assembly"


for sample in "$sample_dir"/*; do
    # Extract the sample name
    sample_name=$(basename "$sample")

    # Make directories for identify and align steps
    mkdir -p "contamination_investigation/identify/$sample_name"
    mkdir -p "contamination_investigation/align/$sample_name"

    # GTDB-Tk identify
    gtdbtk identify --genome_dir "$sample" --out_dir "contamination_investigation/identify/$sample_name" --cpus 20 --write_single_copy_genes --extension fasta

    # GTDB-Tk align
#   gtdbtk align --identify_dir "contamination_investigation/identify/$sample_name" --out_dir "contamination_investigation/align/$sample_name" --taxa_filter "s__Escherichia" --cpus 20
done

module purge


##Write something that subsets the identified genes to the ones identified in the NC
