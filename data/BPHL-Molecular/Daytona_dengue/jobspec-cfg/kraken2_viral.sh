#!/usr/bin/bash
#SBATCH --account=bphl-umbrella
#SBATCH --qos=bphl-umbrella
#SBATCH --job-name=kraken
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48gb
#SBATCH --time=04:00:00
#SBATCH --output=kraken_%j.out
#SBATCH --error=kraken_%j.err

module load apptainer

# generate samples list
ls ./fastqs/*_1.fastq.gz | cut -d'/' -f 3 | sed 's/_1.fastq.gz//' > samples.txt

# identify sample's serotype
mkdir kraken_out_broad

for f in $(cat samples.txt)
do
   echo $f
   #singularity exec -B $(pwd):/data --pwd /data --cleanenv /apps/staphb-toolkit/containers/kraken2_2.1.2-no-db.sif kraken2 --db /blue/bphl-florida/share/kraken_databases/kraken2-broad-custom_other-20200411/ --threads 10 --report kraken_out_broad/${f}.report --output kraken_out_broad/${f}_kraken.out --paired ./fastqs/${f}_1.fastq.gz ./fastqs/${f}_2.fastq.gz
   singularity exec -B $(pwd):/data --pwd /data --cleanenv /apps/staphb-toolkit/containers/kraken2_2.1.2-no-db.sif kraken2 --db /orange/bphl-florida/databases/kraken_databases/kraken2-broad-custom_other-20200411/ --threads 10 --report kraken_out_broad/${f}.report --output kraken_out_broad/${f}_kraken.out --paired ./fastqs/${f}_1.fastq.gz ./fastqs/${f}_2.fastq.gz

done

# move each sample's fastq data files to its serotype folder
#mkdir ./fastqs/dengue1
#mkdir ./fastqs/dengue2
#mkdir ./fastqs/dengue3
#mkdir ./fastqs/dengue4
mkdir ./fastqs/unserotype

echo "Sample","Serotype","Confidence" > Serotypes.txt
for f in $(cat samples.txt)
do
   X=$(grep 'Dengue virus' ./kraken_out_broad/${f}.report | sed '2!d' | rev | cut -d' ' -f 1)
   Y=$(grep 'Dengue virus' ./kraken_out_broad/${f}.report | sed '2!d' | cut -f 1)
   echo ${f},$X,$Y >> Serotypes.txt
   if [ $(echo "${Y}>50.0"|bc -l) -eq 1 ]
   then
      #mv ./fastqs/${f}_1.fastq.gz ./fastqs/dengue${X}/
      #mv ./fastqs/${f}_2.fastq.gz ./fastqs/dengue${X}/
      mv ./fastqs/${f}_1.fastq.gz ./fastqs/SER${X}_${f}_1.fastq.gz
      mv ./fastqs/${f}_2.fastq.gz ./fastqs/SER${X}_${f}_2.fastq.gz
   else
      mv ./fastqs/${f}_1.fastq.gz ./fastqs/unserotype/
      mv ./fastqs/${f}_2.fastq.gz ./fastqs/unserotype/
   fi
done


