#!/bin/bash
#SBATCH -p memory
#SBATCH -N 1 -c 32
#SBATCH -t 120:00:00
#SBATCH -J 101-merge
#SBATCH -A proj5057

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <genome_fasta>"
    echo "Please provide the path to the genome FASTA file as an argument."
    exit 1
fi
genome_fasta=$1
base_name=$(basename "$genome_fasta")
base_name="${base_name%.fasta}"
echo "Processing ${genome_fasta}"

mkdir -p data/repex-output
module load Singularity/3.3.0
singularity exec shub://repeatexplorer/repex_tarean seqclust \
    -p -t -c 120 -v "data/repex-output/${gebase_namenome}" \
    "${genome_fasta}"