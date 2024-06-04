#!/bin/bash
#PBS -N vamb-binning-GPU
#PBS -l nodes=1:ppn=2:gpus=2:exclusive_process
#PBS -l mem=360gb
#PBS -l walltime=48:00:00
#PBS -q inferno
#PBS -A GT-ktk3-CODA20
#PBS -o /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/scripts/log/${PBS_JOBNAME}_${PBS_JOBID}.out
#PBS -e /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/scripts/log/${PBS_JOBNAME}_${PBS_JOBID}.err
#PBS -t 127,131,132,244,245,247,264-266,281,282,284,304,497,539,540,550-552

name=pico${PBS_ARRAYID}
module purge
module load cuda/10.2
source ~/.bashrc
conda init bash
conda activate base
which perl
which samtools
#VARIABLES
name=pico${PBS_ARRAYID}
wd=/storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/01.resample_reads
output=/storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}
contig=/storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}/${name}.contigs.long.fa

~/p-ktk3-0/miniconda3/bin/samtools sort -@ 24 -O bam -o /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}.new.sorted.bam /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}.bam

#### ~/p-ktk3-0/miniconda3/bin/samtools index /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}.sorted.bam /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}.sorted.bam.bai


~/p-ktk3-0/miniconda3/bin/jgi_summarize_bam_contig_depths --outputDepth /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}.new.depth.txt /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/${name}.new.sorted.bam
rm -rf ${output}/out1


~/p-ktk3-0/miniconda3/bin/vamb --outdir ${output}/out1 --fasta /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/catalogue.fna.gz --jgi /storage/home/hcoda1/4/jzhao399/p-ktk3-0/rich_project_bio-konstantinidis/pico_new/02.megahit/*.new.depth.txt -o C --minfasta 200000 --cuda
