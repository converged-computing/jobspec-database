#!/bin/bash -l 

#SBATCH --job-name=Sfa_denovoSSL_100k_mpileup
#SBATCH -o Sfa_denovoSSL_100k_mpileup-%j.out 
#SBATCH --mem 2000 #most runs should be under 1 GB but we'll add a buffer
#SBATCH --array=1-130 # set upper bound to the number of scaffolds that wil be processed 
#SBATCH -p main 
#SBATCH -n 1 #each chromosome will be processed on a single core (-n 1)  
#SBATCH -N 1 #on one machine (-N 1)  
#SBATCH --cpus-per-task=40 
#SBATCH --mail-user=br450@rutgers.edu 
#SBATCH --mail-type=begin 
#SBATCH --mail-type=END 

enable_lmod 

module load container_env ddocent

export SINGULARITY_BIND=/home/e1garcia

CHR="${SLURM_ARRAY_TASK_ID}"
DATAPATH="/home/e1garcia/shotgun_PIRE/2022_PIRE_omics_workshop/salarias_fasciatus/PSMC/data/mkBAM/shotgun_100k"
mkdir -p $DATAPATH/Sfa_denovoSSL_100k_PSMC/
mkdir -p $DATAPATH/Sfa_denovoSSL_100k_PSMC/joblog

crun samtools mpileup -C50 -r $CHR -uf $DATAPATH/reference.denovoSSL.Sfa100k.fasta $DATAPATH/Sfa_denovoSSL_100k.bam | crun bcftools call -c - | crun vcfutils.pl vcf2fq -d 46 -D 276 > $DATAPATH/Sfa_denovoSSL_100k_PSMC/Sfa_denovoSSL_100k.$CHR.fq

mv *mpileup*out $DATAPATH/Sfa_denovoSSL_100k_PSMC/joblog
