#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vst14@case.edu
#SBATCH --mem=185G
#SBATCH -c 20
#SBATCH -C gpu4v100
#SBATCH --time=100:00:00
#SBATCH --job-name=GPU_7043

module purge
module load parabricks/3.1.1 singularity/3.5.1 cuda/10.1 python gatk

cd /scratch/users/vst14/PB_Test/7043_D/

pbrun germline --ref /mnt/rds/txl80/LaframboiseLab/vst14/GRCh37-lite.fa --knownSites /mnt/rds/txl80/LaframboiseLab/sxl919/sxl919/tools/sxl919/b37/dbsnp_138.b37.vcf --in-fq 7043_D_1_R1.fastq.gz 7043_D_1_R2.fastq.gz --in-fq 7043_D_2_R1.fastq.gz 7043_D_2_R2.fastq.gz --out-recal-file 7043_D_Recal.txt --out-bam 7043_D.bam --out-variants 7043_D.g.vcf --gvcf