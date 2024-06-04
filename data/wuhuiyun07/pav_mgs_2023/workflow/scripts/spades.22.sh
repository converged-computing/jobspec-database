#!/bin/bash
#SBATCH --job-name=pav-mgs 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out-%N  
#SBATCH --error=slurm-%j.err-%N 
#SBATCH --account=loni_virus2023      # your account name
#SBATCH --partition=workq             # the partition

# export WORK_DIR=/project/awlab/wuhuiyun/pav_mgs_2023
# Load any required modules for your HPC.

export OMP_NUM_THREADS=48

module load snakemake
module load python
module load spades/3.13.0/gcc-9.3.0
# Run snakemake
# snakemake --profile config/slurm --latency-wait 90 --use-singularity --use-conda --configfile config/test.yaml
# cd /project/awlab/wuhuiyun/pav_mgs_2023
export WORK_DIR=/project/awlab/wuhuiyun/pav_mgs_2023

# SAMPLES = "16_1_S1 16_2_S2 16_3_S3 16_4_S4 16_5_S5 22_1_S6 22_2_S7 22_3_S8 22_4_S9 22_5_S10 23_1_S11 23_2_S12 23_3_S13 23_4_S14 23_5_S15 24_1_S16 24_2_S17 24_3_S18 24_4_S19 24_5_S20 29_1_S21 29_2_S22 29_3_S23 29_4_S24 29_5_S25 Blank_8_15_23_S26".split()

 spades.py --version
        export OMP_NUM_THREADS=48
        spades.py --meta \
            --threads 48 \
            -1 results/trimmed/22_1_S6.R1.fastq.gz \
            -2 results/trimmed/22_1_S6.R2.fastq.gz \
            -o results/assembly/test/22_1_S6/

 spades.py --version
        export OMP_NUM_THREADS=48
        spades.py --meta \
            --threads 48 \
            -1 results/trimmed/22_2_S7.R1.fastq.gz \
            -2 results/trimmed/22_2_S7.R2.fastq.gz \
            -o results/assembly/test/22_2_S7/
            
 spades.py --version
        export OMP_NUM_THREADS=48
        spades.py --meta \
            --threads 48 \
            -1 results/trimmed/22_3_S8.R1.fastq.gz \
            -2 results/trimmed/22_3_S8.R2.fastq.gz \
            -o results/assembly/test/22_3_S8/
             spades.py --version
        export OMP_NUM_THREADS=48

spades.py --meta \
    --threads 48 \
    -1 results/trimmed/22_4_S9.R1.fastq.gz \
    -2 results/trimmed/22_4_S9.R2.fastq.gz \
    -o results/assembly/test/22_4_S9/

 spades.py --version
        export OMP_NUM_THREADS=48
        spades.py --meta \
            --threads 48 \
            -1 results/trimmed/22_5_S10.R1.fastq.gz \
            -2 results/trimmed/22_5_S10.R2.fastq.gz \
            -o results/assembly/test/22_5_S10/

    