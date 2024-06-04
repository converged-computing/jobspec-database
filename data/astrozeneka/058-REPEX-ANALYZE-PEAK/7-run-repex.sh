#!/bin/bash
#SBATCH -p memory
#SBATCH -N 1 -c 40
#SBATCH -t 120:00:00
#SBATCH -J analyze-peak-betta
#SBATCH -A proj5057

module load Singularity/3.3.0

genomes=(
  "SRR7062760"
)

cd data/raw_map
for genome in "${genomes[@]}"
do
  singularity exec shub://repeatexplorer/repex_tarean seqclust \
      -p -t -c 64 -v "repex-${genome}" \
      "data/raw_map/${genome}_mapped.fasta"
  echo "Done"
done