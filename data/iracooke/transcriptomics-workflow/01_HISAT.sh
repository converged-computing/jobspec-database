#!/bin/bash
#PBS -c s
#PBS -j oe
#Send a mail alert at abortion, beginning and end of execution
#PBS -m abe
#Set the name of the job
#PBS -N hisat2
#Send any mail to this address
#PBS -M username@my.jcu.edu.au
#Allocate required amount of wall time
#PBS -l walltime=1000:00:00
#Set the number of nodes and processors
#PBS -l nodes=1:ppn=4
#Allocate required amount of memory
#PBS -l pmem=4gb

#Make DUMMY the variable for all files. This may then be substituted with the name of a file in the command loop
f=DUMMY

#Navigate into the desired directory using an absolute pathway
cd /shares/32/jc320251/untouched_data

#Present path to hisat2 module and specify the number of processors [as above];
#Present path to the genome index - relative to the directory previously listed and;
#Align the files and output them as .sam files
/sw/hisat2/2.0.5/hisat2 -p 4 -x ../grch38/genome -U $f -S ${f%.fastq.gz}.sam;
#Present the path to the samtools module
#Sort the data and;
#Output to bam file
/sw/samtools/1.3/AMD/bin/samtools sort -@ 4 -o ${f%.fastq.gz}.bam ${f%.fastq.gz}.sam;
