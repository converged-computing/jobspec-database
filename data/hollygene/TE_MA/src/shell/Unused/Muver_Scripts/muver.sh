#PBS -S /bin/bash
#PBS -N muverTestJuly19
#PBS -q batch
#PBS -l nodes=1:ppn=48:AMD
#PBS -l walltime=480:00:00
#PBS -l mem=100gb
#PBS -M hcm14449@uga.edu
#PBS -m abe


cd $PBS_O_WORKDIR
#location of current update of muver
muver_module="muver/0.1.0-foss-2016b-Python-2.7.14-20190318"
#location of trimgalore moedule
# trimgalore_module="Trim_Galore/0.4.5-foss-2016b"
#location of data to be analyzed
# data_dir="/scratch/hcm14449/TE_MA_Paradoxus/test_Spike_InsJune2019/Holly_spikein_289/Holly_spikein"
# anc_dir="/scratch/hcm14449/TE_MA_Paradoxus/test_Spike_InsJune2019/Holly_gDNA/"
# #link to reference genome to download
# ref_gen_download="https://downloads.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_Current_Release.tgz"
# #location of reference genome to be used
# ref_genome="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/YPS138.genome.fa"
# #directory reference genome is located in
# ref_genome_dir="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus"
# #text file listing the fastq files with their full extensions
# h0_fastq_list="/home/hcm14449/Github/TE_MA/H0_FASTQ_LIST.txt"
#what sample should all other samples be compared to?
# h0_control_sample_name="H0_ANC"
# d0_fastq_list="/home/hcm14449/Github/TE_MA/D0_FASTQ_LIST.txt"
# d0_control_sample_name="D0_ANC"
# d1_fastq_list="/home/hcm14449/Github/TE_MA/D1_FASTQ_LIST.txt"
# d1_control_sample_name="D1_ANC"
# d20_fastq_list="/home/hcm14449/Github/TE_MA/D20_FASTQ_LIST.txt"
# d20_control_sample_name="D20_ANC"
#where should the output be sent
# h0_experiment_directory="/scratch/hcm14449/TE_MA_Paradoxus/Practice/Test_RunJuly19/H0"
# mkdir $h0_experiment_directory
# d0_experiment_directory="/scratch/hcm14449/TE_MA_Paradoxus/Practice/Test_RunJuly19/D0"
# mkdir $d0_experiment_directory
# d1_experiment_directory="/scratch/hcm14449/TE_MA_Paradoxus/Practice/Test_RunJuly19/D1"
# mkdir $d1_experiment_directory
# d20_experiment_directory="/scratch/hcm14449/TE_MA_Paradoxus/Practice/Test_RunJuly19/D20"
# mkdir $d20_experiment_directory
#location of TRIMMED data to be used in the analysis
# trimmed_data="/scratch/hcm14449/TE_MA_Paradoxus/Practice/files/samples/trimmed"

data_dir="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/IL_Data/GW_run3/00_fastq"
ref_genome="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/YPS138.genome.fa"
# fastq_list="/home/hcm14449/Github/TE_MA/FASTQ_LIST.txt"
# control_sample_name="Ancestor"
# experiment_directory="/scratch/hcm14449/TE_MA_Paradoxus/Practice/output8919"

fastq_list_H0="/home/hcm14449/Github/TE_MA/FASTQ_LIST_H0.txt"
control_sample_name_H0="H0-A"
experiment_directory_H0="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Muver/H0"
mkdir $experiment_directory_H0

fastq_list_D0="/home/hcm14449/Github/TE_MA/FASTQ_LIST_D0.txt"
control_sample_name_D0="D0-A"
experiment_directory_D0="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Muver/D0"
mkdir $experiment_directory_D0

fastq_list_D1="/home/hcm14449/Github/TE_MA/FASTQ_LIST_D1.txt"
control_sample_name_D1="D1-A"
experiment_directory_D1="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Muver/D1"
mkdir $experiment_directory_D1

fastq_list_D20="/home/hcm14449/Github/TE_MA/FASTQ_LIST_D20.txt"
control_sample_name_D20="D20-A"
experiment_directory_D20="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Muver/D20"
mkdir $experiment_directory_D20

# mkdir $trimmed_data
#
# module load ${trimgalore_module}
#
# # trim all fastq files
# for file in $data_dir/*.fastq
#
# do
#
# FBASE=$(basename $file .fastq)
# BASE=${FBASE%.fastq}
#
# trim_galore --phred33 -q 20 -o $trimmed_data ${BASE}.fastq
#
# done
#
# module unload ${trimgalore_module}

# cd $ref_genome_dir
# download the desired reference genome for analysis
# wget $ref_gen_download


module load ${muver_module}
# module load ${GATK_module}
# java -jar $EBROOTGATK/GenomeAnalysisTK.jar

#index reference genome
# echo ${ref_genome}
# muver index-reference ${ref_genome}
# # muver index-reference S288C_reference_sequence_R64-2-1_20150113.fa
# #
# # create repeat file for reference genome
# muver create-repeat-file ${ref_genome}

muver run-pipeline -p 48 ${ref_genome} ${fastq_list_H0} ${control_sample_name_H0} ${experiment_directory_H0}

muver run-pipeline -p 48 ${ref_genome} ${fastq_list_D0} ${control_sample_name_D0} ${experiment_directory_D0}

muver run-pipeline -p 48 ${ref_genome} ${fastq_list_D1} ${control_sample_name_D1} ${experiment_directory_D1}

muver run-pipeline -p 48 ${ref_genome} ${fastq_list_D20} ${control_sample_name_D20} ${experiment_directory_D20}
# run the pipeline
# muver run-pipeline ${ref_genome} ${h0_fastq_list} ${h0_control_sample_name} ${h0_experiment_directory}
#
# muver run-pipeline ${ref_genome} ${d0_fastq_list} ${d0_control_sample_name} ${d0_experiment_directory}
#
# muver run-pipeline ${ref_genome} ${d1_fastq_list} ${d1_control_sample_name} ${d1_experiment_directory}
#
# muver run-pipeline ${ref_genome} ${d20_fastq_list} ${d20_control_sample_name} ${d20_experiment_directory}


# module load muver/0.1.0-foss-2016b-Python-2.7.14-20190318
# # java -jar $EBROOTGATK/GenomeAnalysisTK.jar
# # java -jar $EBROOTPICARD/picard.jar
# muver run-pipeline /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/YPS138.genome.fa /home/hcm14449/Github/TE_MA/H0_FASTQ_LIST.txt H0_ANC /scratch/hcm14449/TE_MA_Paradoxus/Practice/Test_RunJuly19/H0

# muver create-repeat-file /scratch/hcm14449/TE_MA_Paradoxus/Practice/files/samples/Sample2_R1.fastq
#
# muver fit-repeat-indel-rates /scratch/hcm14449/TE_MA_Paradoxus/Practice/output.3/bams/Sample2.bam

# muver run-pipeline /scratch/hcm14449/TE_MA_Paradoxus/Practice/files/ref_genome/SCerevisiae.RefGenome.fa.fai \
# /home/hcm14449/Github/TE_MA/FASTQ_LIST.txt Ancestor /scratch/hcm14449/TE_MA_Paradoxus/Practice/output
