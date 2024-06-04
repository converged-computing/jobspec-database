#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=8,vmem=100gb,walltime=24:00:00
#PBS -M wrshoema@iu.edu
#PBS -m abe
#PBS -j oe
module load python
module load gcc
#module load cutadapt
module load java
module load fastqc

#-g CAAGCAGAAGACGGCATACGA
#-g AATGATACGGCGACCACCGA

Sample_date=D100
#Sample_date=${var1}
# fastqc of raw data
mkdir -p /N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_quality/
mkdir -p "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_quality/${Sample_date}/"

for file in "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_raw/${Sample_date}/"*/*.fastq.gz
do
  sample="$(echo "$file" | cut -d "/" -f11-11)"
  OUTdir="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_quality/${Sample_date}/${sample}"
  mkdir -p $OUTdir
  fastqc "$file" --outdir=$OUTdir
done

# trim data and remove adaptors.
mkdir -p "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic"
mkdir -p "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic/${Sample_date}"

for folder in "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_raw/${Sample_date}/"*/
do
  declare -a ARRAYlines=()
  for file in $folder*.fastq.gz
  do
    if [[ $file == *"_R1_"* ]]; then
      line="$(  echo "$file" | cut -d"/" -f11-13)"
      ARRAYlines=("${ARRAYlines[@]}" "$line")
    fi
  done
  line_folder="$(  echo "$folder" | cut -d"/" -f11-12)"
  mkdir -p "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic/${Sample_date}/${line_folder}"
  for ARRAYline in "${ARRAYlines[@]}"
  do
    R1=$ARRAYline
    R2="${R1/_R1_/_R2_}"
    InR1="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_raw/${Sample_date}/${R1}"
    InR2="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_raw/${Sample_date}/${R2}"
    OutR1PairedName="${R1/.fastq.gz/_clean_paired.fastq.gz}"
    OutR2PairedName="${R2/.fastq.gz/_clean_paired.fastq.gz}"
    OutR1UnPairedName="${R1/.fastq.gz/_clean_unpaired.fastq.gz}"
    OutR2UnPairedName="${R2/.fastq.gz/_clean_unpaired.fastq.gz}"
    OutR1Paired="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic/${Sample_date}/${OutR1PairedName}"
    OutR2Paired="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic/${Sample_date}/${OutR2PairedName}"
    OutR1UnPaired="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic/${Sample_date}/${OutR1UnPairedName}"
    OutR2UnPaired="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic/${Sample_date}/${OutR2UnPairedName}"
    adaptor1="$(  echo "$OutR1Paired" | cut -d"_" -f5-5 | cut -d"-" -f1-1)"
    adaptor2="$(  echo "$OutR1Paired" | cut -d"_" -f5-5 | cut -d"-" -f2-2)"
    java -jar /N/dc2/projects/MicroEukMA/softwares/Trimmomatic-0.32/trimmomatic-0.32.jar \
      PE -threads 4 $InR1 $InR2 $OutR1Paired $OutR1UnPaired $OutR2Paired $OutR2UnPaired \
      ILLUMINACLIP:/N/dc2/projects/muri2/Task2/PoolPopSeq/bin/transposase.fa:2:30:10 \
      LEADING:4 TRAILING:4 MINLEN:40 HEADCROP:15
  done
done

# fastqc of trimmed data
mkdir -p "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic_quality/"
mkdir -p "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic_quality/${Sample_date}/"

for file in "/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic/${Sample_date}/"*/*.fastq.gz
do
  sample="$(echo "$file" | cut -d "/" -f11-11)"
  OUTdir="/N/dc2/projects/muri2/Task2/PoolPopSeq/data/reads_clean_trimmomatic_quality/${Sample_date}/${sample}"
  mkdir -p $OUTdir
  fastqc "$file" --outdir=$OUTdir
done
