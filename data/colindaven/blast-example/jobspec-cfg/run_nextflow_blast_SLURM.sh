#!/bin/bash
## Supply the FASTA input as arg1, sbatch run_nextflow_blast_SLURM.sh in.fasta


# set partition
#SBATCH -p old

# set run on all mem nodes
#SBATCH --mem 15000

# set run on bigmem node only
#SBATCH --cpus-per-task 1

# share node
#SBATCH --share

# set max wallclock time
#SBATCH --time=300:00:00

# set name of job
#SBATCH --job-name=blast_start

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=<your_email>


echo "Input file: " $1
fasta=$1

PWD="pwd"
NXF_ASSETS="$PWD/$fasta.assets"
NXF_TEMP="$PWD/$fasta.temp"
NXF_WORK="$PWD/$fasta.work"

# Add miniconda3 to PATH
. /mnt/ngsnfs/tools/miniconda3/etc/profile.d/conda.sh

# Activate env on cluster node
#conda activate PPKC_env

blastdb=/lager2/rcug/seqres/nt_db/nt

# Run script
nextflow /ngsssd1/rcug/nextflow_blast/main3.nf -c /ngsssd1/rcug/nextflow_blast/nextflow1.conf --query $fasta --db $blastdb --chunkSize 100 -with-report $fasta.report.html -with-timeline $fasta.timeline.html -with-trace > $1.csv
