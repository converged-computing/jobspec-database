#!/bin/bash

# set the number of nodes
#SBATCH --ntasks=8
#SBATCH --partition=amd
# set max wallclock time
#SBATCH --time=296:00:00

#mail alert at start,end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=renxx275@umn.edu

#run the application
cd $SLURM_SUBMIT_DIR
sh /usr/local/modules/profile.modules
# load the hisat2 module
#module load hisat2
#module load samtools
#module load stringtie
#module load subread
#Mapping the reads with hisat

source /atavium/home/yang4414/renxx275/anaconda3/etc/profile.d/conda.sh
conda activate python2

for each in `cat fastq.txt`
do
hisat2 -p 8 -x /atavium/home/yang4414/cleanup/reference_genomes/hg38/hisat2/genome -U ${each}.fastq | /usr/bin/samtools view -Sbo ${each}.bam -
/usr/bin/samtools sort ${each}.bam -o ${each}_sorted.bam
/usr/bin/samtools index ${each}_sorted.bam
done


#featureCounts -p -t exon -g gene_id -a /atavium/home/yang4414/tywang/projects/lncRNAs/lncRNAs_data/SU2C_PolyA/noval_lncRNA/ref_w_novel.hg38.gtf -o release_id_counts_test.txt /home/ryang/project/k562_drug/release1_IN_sorted.bam /home/ryang/project/k562_drug/release2_IN_sorted.bam
