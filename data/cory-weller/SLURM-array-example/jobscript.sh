#!/usr/bin/env bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --time 2:00:00
#SBATCH --mem 20G
#SBATCH --partition norm

# define file that contains job array paremeters
PARAMETER_FILE='sample_ids.txt'
BWA_INDEX='path/to/index_base'

# extract Nth line from PARAMETER_FILE and save it as
# the variable named SAMPLE
SAMPLE=$(sed -n ${SLURM_ARRAY_TASK_ID}p ${PARAMETER_FILE})

# Define fastq file names based on this SAMPLE variable
FASTQ_1="data/${SAMPLE}_R1_001.fastq.gz"
FASTQ_2="data/${SAMPLE}_R2_001.fastq.gz"

# Do whatever needs to be done with your inputs
module load bwa
bwa mem ${BWA_INDEX} $ ${FASTQ_1} ${FASTQ_2} > ${SAMPLE}.sam