#!/bin/bash
#SBATCH --array=0-3 ##specify number of samples
#SBATCH --output=%A_%a.o
#SBATCH --error=%A_%a.e
#SBATCH --job-name=%a_rnaseq-array-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --mail-user=${USER}@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load trim_galore/0.4.4
module load cutadapt/intel/1.12
module load fastqc/0.11.5
module load hisat2/intel/2.0.5
module load samtools/intel/1.3.1
module load r/intel/3.3.2
module load umi_tools/0.4.2
module load qualimap/2.2.1
module load gffread/intel/0.9.8c
module load biopython/intel/1.70

#Configuration
analysisName="test_that_se_umi"
isPE=0 #0 if SE
isUMI=1 #1 if UMIs used

#File locations
#DIR='/scratch/cgsb/gresham/LABSHARE/Scripts/Pipelines/SE_UMI_RNAseq_gz/SE_UMI_RNAseq_head'
#DIR='/scratch/cgsb/gencore/out/Gresham/2014-09-24_HAJW7ADXX/umi/1'
#DIR='/scratch/cgsb/gencore/out/Gresham/2015-03-05_C572CACXX/umi/4'
DIR='/scratch/cgsb/gresham/LABSHARE/Scripts/Pipelines/testdata_SE_UMI'
hisat_ref='/scratch/work/cgsb/genomes/In_house/Fungi/Saccharomyces_cerevisiae/Gresham/GCF_000146045.2_R64_GreshamLabSpikes/GCF_000146045.2_R64_GreshamLabSpikes_genomic_hisat_index/genome'
#Below is the directory	of the custom gff
gff='/scratch/work/cgsb/genomes/In_house/Fungi/Saccharomyces_cerevisiae/Gresham/GCF_000146045.2_R64_GreshamLabSpikes/GCF_000146045.2_R64_GreshamLabSpikes_genomic.gff'
#Below is the directory of the standard gff
#gff='/genomics/genomes/Public/Fungi/Saccharomyces_cerevisiae/NCBI/R64/GCF_000146045.2_R64_genomic.gff'



#Make a local directory and out-of-stream resource copies of this script for records
mkdir -p ${analysisName}
mkdir -p ${analysisName}/logs
cp rnaseq-array-pipeline.sh "${analysisName}/logs/rnaseq-array-pipeline.$(date +%F_%R).sh"
cp ${Read1_Adapter} ${analysisName}/logs
cp ${Read2_Adapter} ${analysisName}/logs
cd ${analysisName}

echo "Generating the gtf file from the gff, using gffread"
time gffread -E ${gff} -T -o- > temp.gtf
gtf1='temp.gtf'

## list of input files
#FILES=($(ls $DIR/*fastq.gz | perl -pe 's/^.+n0\d_(.+)\.fastq\.gz/$1/g' | sort | uniq))
FILES=($(ls $DIR/*fastq | perl -pe 's/^.+n0\d_(.+)\.fastq/$1/g' | sort | uniq))

echo "Processing these samples: ${FILES[*]}"

## $ID will hold the sample_id
ID=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "This is the FILE I'm running"
echo ${FILES}
echo "This is it's ID"
echo ${ID}
echo "This is the ARRAY number I've been assigned"
echo ${SLURM_ARRAY_TASK_ID}

echo "I'm making this directory and jumping into it: ${ID}"
mkdir $ID
cd $ID

#Update gtf variable as it only exists once and is in analysis directory
gtf="../"${gtf1}

echo "I think my GTF is at ${gtf}" 

#############
# Alignment #
#############

if [ "${isPE}" == 1 ] ; then

  ## full paths to paired mate files
  INPUT_1=$(ls $DIR/*n01_$ID.fastq)
  INPUT_2=$(ls $DIR/*n02_$ID.fastq)

  if [ "${isUMI}" = 0 ] ; then

    echo "Now I'm trimming the reads for adapters, quality?"
    time trim_galore --paired --fastqc $INPUT_1 $INPUT_2
    time hisat2 -p ${SLURM_CPUS_ON_NODE} -q -x ${hisat_ref} -1 *n01_${ID}_val_1.fq.gz -2 *n02_${ID}_val_2.fq.gz -S ${ID}.sam
  
  elif [ "${isUMI}" = 1 ] ; then

    echo "I assume you're using the trumiseq adapters, so trimming"
    time python ../../scrape_adapter.py -u -i $INPUT_1 -o Read1_Adapter.fa
    time python ../../scrape_adapter.py -p -i $INPUT_2 -o Read2_Adapter.fa
    time cutadapt -a file:Read1_Adapter.fa -A file:Read2_Adapter.fa -o ${ID}_1.trim -p ${ID}_2.trim $INPUT_1 $INPUT_2
    time fastqc -o ${ID}_postTrimming -f fastq ${ID}_1.trim ${ID}_2.trim
    time hisat2 -p ${SLURM_CPUS_ON_NODE} -q -x ${hisat_ref} -1 ${ID}_1.trim -2 ${ID}_2.trim -S $ID.sam
  
  fi

  echo "Okay, now I sort and index that bam"
  time samtools sort -o ${ID}.bam ${ID}.sam
  time samtools index ${ID}.bam

  echo "And delete that file, and fastqz files as well."
  rm ${ID}.sam

  if [ "${isUMI}" = 0 ] ; then

    rm *n01_${ID}_val_1.fq.gz
    rm *n02_${ID}_val_2.fq.gz
  
  elif [ "${isUMI}" = 1 ] ; then

    rm ${ID}_1.trim
    rm ${ID}_2.trim

  fi

  if [ "${isUMI}" = 1 ] ; then

    echo "You want me to dedup? Okay, deduping with umi_tools, then 
    sorting and deleting the unsorted bam."
    time umi_tools dedup --umi-separator=: --method=directional --output-stats=${ID}.stats -I ${ID}.bam -S ${ID}.dedup.bam -L ${ID}.dedup.log
    time samtools sort -o ${ID}.sorted.dedup.bam ${ID}.dedup.bam
    rm ${ID}.dedup.bam
    
    ##Determine counts                                                                                                                                         
    echo "There were "$( samtools view ${ID}.bam | wc -l )" counts before umi-deduplication"
    echo "There are "$( samtools view ${ID}.sorted.dedup.bam | wc -l )" counts after umi-deduplication"
  
    echo "Deleting a lot of intermediate files, indexing."
    rm ${ID}.bam
    rm ${ID}.bam.bai
    mv ${ID}.sorted.dedup.bam ${ID}.bam
    time samtools index ${ID}.bam

  fi

elif [ "${isPE}" = 0 ] ; then

  ## full paths to single end files
  INPUT_1=$(ls $DIR/*n01_$ID.fastq)

  if [ "${isUMI}" = 0 ] ; then

    echo "Trimgaloring that ${INPUT_1}, then aligning"
    time trim_galore --fastqc $INPUT_1
    time hisat2 -q -x ${hisat_ref} -1 $INPUT_1 $ID.sam

  elif [ "${isUMI}" = 1 ] ; then

    echo "Trimming that ${INPUT_1} for adapters, quality? aligning"
    echo ${ID}_1.trim
    time python ../../scrape_adapter.py -u -i $INPUT_1 -o Read1_Adapter.fa
    time cutadapt -a file:Read1_Adapter.fa -o ${ID}_1.trim $INPUT_1
    time hisat2 -q -x ${hisat_ref} -U ${ID}_1.trim -S $ID.sam

  fi

  echo "sort and index"
  time samtools sort -o ${ID}.bam ${ID}.sam
  time samtools index ${ID}.bam

  #Clean up .sam and .fastq files
  rm ${ID}.sam

  if [ "${isUMI}" = 1 ] ; then
    rm ${ID}_1.trim
    
    echo "and doing some deduping"
    time umi_tools dedup --umi-separator=: --method=directional --output-stats=${ID}.stats -I ${ID}.bam -S ${ID}.dedup.bam -L ${ID}.dedup.log
    time samtools sort -o ${ID}.sorted.dedup.bam ${ID}.dedup.bam
    rm ${ID}.dedup.bam
  
    ##Determine counts 
    echo "There were "$( samtools view ${ID}.bam | wc -l )" counts before umi-deduplication"
    echo "There are "$( samtools view ${ID}.sorted.dedup.bam | wc -l )" counts after umi-deduplication"
  
    ##Clean up non-deduplicated .bam file
    rm ${ID}.bam
    rm ${ID}.bam.bai
    mv ${ID}.sorted.dedup.bam ${ID}.bam
    samtools index ${ID}.bam

  fi

else
 
  echo "Define analysis as either a paired end or single end sequencing protocol"

fi

##############
# QC metrics #
##############

#Run Qualimap on individual BAM file
#http://qualimap.bioinfo.cipf.es/doc_html/command_line.html#command-line
#NB. for qualimap rnaseq set -pe if paired end sequencing used 
#Java virtual machine uses DISPLAY environment variable to detect if the X11 system is available. Sometimes this variable is set incorrectly by the operating system or some applications. To make Qualimap work simply unset this variable:
echo "Now doing qualimaps"
unset DISPLAY
if [ "${isUMI}" = 0 ] ; then

  time qualimap bamqc -bam ${ID}.bam -gff ${gff} -c -outdir ${ID}_stats -outfile ${ID}
  time qualimap rnaseq -bam ${ID}.bam -gtf ${gtf} -pe -outdir rnaseq_qc_results -outfile ${ID}

elif [ "${isUMI}" = 1 ] ; then

  time qualimap bamqc -bam ${ID}.bam -gff ${gff} -c
  time qualimap rnaseq -bam ${ID}.bam -gtf ${gtf} -outdir rnaseq_qc_results

fi

############################
# Generate table of counts #
############################

#Call Rsubread

echo "and doing rsubread"
time Rscript --vanilla ../../feature_counts.R ${gtf1} ${isPE} > ../logs/rsubread.log

echo "Completed now moving the STDOUT and STDERR logs from sbatch
into the above directory."

#################################################
# Move output and error files to logs directory #
#################################################

mv ../../${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.o ../logs
mv ../../${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.e ../logs

exit 0;
