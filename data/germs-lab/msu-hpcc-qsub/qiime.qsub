#!/bin/bash -login

### define resources needed:
### walltime - how long you expect the job to run
#PBS -l walltime=5:00:00:00
 
### nodes:ppn - how many nodes & cores per node (ppn) that you require
#PBS -l nodes=20:ppn=1

### mem: amount of memory that the job will need
#PBS -l mem=200gb

#PBS -q main
#PBS -M genase23@gmail.com
#PBS -m abe

### you can give your job a name for easier identification
#PBS -N qiime

### load module
module load pandaseq
module load git
module load Python/2.7.3 
export PATH=$PATH:/mnt/home/choiji22/vsearch-2.3.3/bin
export PATH=$PATH:/mnt/home/choiji22/

module load eaUtils/1.1.2.7-537

### Change directory
cd your_directory

### call

## download database for chimera removal
wget http://greengenes.lbl.gov/Download/Sequence_Data/Fasta_data_files/Caporaso_Reference_OTUs/gg_otus_4feb2011.tgz
tar -xvzf gg_otus_4feb2011.tgz

## in case, mapping file have window's return carriage
tr '\15' '\n' < mapping.txt > mapping.unix.txt

## If you need demultiplex,
git clone https://github.com/metajinomics/qiime_tools.git
python qiime_tools/demultiplex_sequences.py -m mapping.unix.txt -b Undetermined_S0_L001_I1_001.fastq.gz -f Undetermined_S0_L001_R1_001.fastq.gz -r Undetermined_S0_L001_R2_001.fastq.gz -o demultiplexed

## merge using Pandaseq
mkdir merged                                                                                                               
for x in demultiplexed/*R1.fastq;do filename=${x##*/};echo "pandaseq -f $x -r ${x%R1*}R2.fastq -u ${x%R1*}unmerged.fa 2> ${x%R1*}pandastat.txt 1> merged/${filename%R1*}fasta";done > command.panda.sh                                                
cat command.panda.sh | /mnt/home/choiji22/parallel-20161122/src/parallel   

##prepare mapping file                                                                                                      
python qiime_tools/add_filename_into_mapping_file.py mapping_corrected.txt > mapping_filename_added.txt

## start virtual environment for running qiime
source /mnt/home/choiji22/miniconda3/bin/activate qiime1

## if you want to merge and demuliplex using Qiime's function, then use below
## merge paired end
#join_paired_ends.py -f Undetermined_S0_L001_R1_001.fastq.gz -r Undetermined_S0_L001_R2_001.fastq.gz -o fastq-join_joined -b Undetermined_S0_L001_I1_001.fastq.gz

## demultiplex
#validate_mapping_file.py -m mapping.txt
#split_libraries_fastq.py -i fastq-join_joined/fastqjoin.join.fastq -b fastq-join_joined/fastqjoin.join_barcodes.fastq -o out_q20/ -m mapping_corrected.txt -q 19 --rev_comp_mapping_barcodes --store_demultiplexed_fastq
#split_libraries_fastq.py -i fastq-join_joined/fastqjoin.join.fastq -b fastq-join_joined/fastqjoin.join_barcodes.fastq -o out_q20/ -m mapping_corrected.txt -q 19 --store_demultiplexed_fastq

## combine sequences                                                                                                        
add_qiime_labels.py -i merged/ -m  mapping_filename_added.txt -c InputFileName

## remove chimera
identify_chimeric_seqs.py -i combined_seqs.fna -m usearch61 -o usearch_checked_chimeras/ -r gg_otus_4feb2011/rep_set/gg_97_otus_4feb2011.fasta 
filter_fasta.py -f combined_seqs.fna -o seqs_chimeras_filtered.fna -s usearch_checked_chimeras/chimeras.txt -n

## run qiime pipeline
pick_open_reference_otus.py -i seqs_chimeras_filtered.fna -o uclust_openref/
