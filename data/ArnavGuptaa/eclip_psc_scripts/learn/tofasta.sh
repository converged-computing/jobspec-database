#!/bin/bash

#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --no-requeue

set -x

GENOME=/path/to/genome/hg38.fa 

# echo $GENOME

module load anaconda3 
module load bedtools
conda activate pytorch 

echo "input bed: " $1

python ./tofasta.py --bed $1

bedtools getfasta \
-fi /media/alvin/Elements/genomes/hg38/Homo_sapiens_assembly38.fasta \
-bed ./temp.bed > $2

rm ./temp.bed

conda deactivate