#!/bin/bash
#SBATCH --job-name align_reads
#SBATCH --time 240:00:00
#SBATCH --mail-type ALL
#SBATCH --nodes 1
#SBATCH --ntasks 20
#SBATCH --array 1-36
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --account=leachlj-potato-qtl-project

set -e
module purge; module load bluebear
module load STAR/2.7.2b-GCC-8.3.0

DATABASE="/rds/projects/l/leachlj-potato-qtl-project/SharedGenomeFilesV6"
INPUT_DIRECTORY="/rds/projects/l/leachlj-potato-qtl-project/M6_Area/potato-blight-rna-seq-pipeline/trimmed"
OUTPUT_DIRECTORY="/rds/projects/l/leachlj-potato-qtl-project/M6_Area/potato-blight-rna-seq-pipeline/aligned"

if [ ! -d "${OUTPUT_DIRECTORY}" ]; then
	mkdir -p "${OUTPUT_DIRECTORY}"
fi

if [[ "${SLURM_ARRAY_TASK_ID}" -le 18 ]]; then
	SAMPLE_PREFIX="R"
else
	SAMPLE_PREFIX="A"
fi

SAMPLE="${SAMPLE_PREFIX}${SLURM_ARRAY_TASK_ID}"

STAR \
--genomeDir "${DATABASE}/2.STAR-indexV6" \
--readFilesIn "${INPUT_DIRECTORY}/${SAMPLE}_1_val_1.fq.gz" "${INPUT_DIRECTORY}/${SAMPLE}_2_val_2.fq.gz" \
--outFileNamePrefix "${OUTPUT_DIRECTORY}/${SAMPLE}_" \
--runThreadN "${SLURM_NTASKS}" \
--readFilesCommand zcat \
--outSAMunmapped Within \
--alignIntronMax 5000 \
--outSAMtype BAM SortedByCoordinate \
--quantMode TranscriptomeSAM
