#!/bin/bash -l
#SBATCH -A sens2017538
#SBATCH -J Scoring
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 8:00:00
#SBATCH -a 0-175

module load bioinfo-tools plink/1.90b4.9 

let "chr = $SLURM_ARRAY_TASK_ID % 22 + 1"
let "file = $SLURM_ARRAY_TASK_ID / 22"

# Get the weights file we're going to work with
all_weight_files=(weights/*)
weight_file=${all_weight_files[$file]}

# Get the basename of the file for reporting and writing to the correct path
weight_basename=$(basename $weight_file)
echo "Scoring chromosome $chr with weights from $weight_basename."

# Create output directory
# -p flag ensures that all parent directories are created and the script doesn't fail if the path already exists.
mkdir -p scores/$weight_basename

plink \
  --bed genotypes/chr${chr}.bed \
  --bim genotypes/chr${chr}.dedup.bim \
  --fam sample_info/samples.fam \
  --score $weight_file 3 5 7 header sum \
  --keep sample_info/validation.ids \
  --out scores/$weight_basename/scores_chr${chr}
