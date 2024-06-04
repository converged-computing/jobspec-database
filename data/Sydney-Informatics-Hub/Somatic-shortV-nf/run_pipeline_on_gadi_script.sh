#!/bin/bash
  
#PBS -P er01 
#PBS -N Somatic-shortV
#PBS -l walltime=06:00:00
#PBS -l ncpus=1
#PBS -l mem=60GB
#PBS -W umask=022
#PBS -q copyq
#PBS -e Somatic-shortV-nf.e
#PBS -o Somatic-shortV-nf.o
#PBS -l wd
#PBS -l storage=scratch/er01+gdata/er01
#PBS -l jobfs=10GB

# Load singularity and nextflow modules
module load nextflow/22.04.3
module load singularity

export NXF_SINGULARITY_CACHEDIR=/scratch/$PROJECT/$(whoami)/singularity

# Fill in these variables for your run
samples=/scratch/er01/ndes8648/pipeline_work/nextflow/INFRA-83-Somatic-ShortV/test_files_for_Georgie/samples.csv
#samples=/scratch/er01/ndes8648/pipeline_work/nextflow/INFRA-83-Somatic-ShortV/test_files_for_Georgie/samples_full.csv
ponvcf=/scratch/er01/ndes8648/pipeline_work/nextflow/INFRA-83-Somatic-ShortV/test_files_for_Georgie/pon.vcf.gz
ref=/g/data/er01/SIH-HPC-WGS/Reference/hs38DH.fasta
common_biallelic_variants=/g/data/er01/SIH-HPC-WGS/Reference/gatk-best-practices/somatic-hg38/small_exac_common_3.hg38.vcf.gz
outDir=results
dict=/g/data/er01/SIH-HPC-WGS/Reference/hs38DH.dict
#number_of_intervals=23


# Run the pipeline 
nextflow run main.nf -resume \
        --input ${samples} \
        -profile gadi \
        --whoami $(whoami) --gadi_account $PROJECT \
        --ref ${ref} --dict ${dict} \
	--common_biallelic_variants ${common_biallelic_variants} \
        --ponvcf ${ponvcf} \
        --outDir ${outDir} \
        --number_of_intervals ${number_of_intervals}
        

        
