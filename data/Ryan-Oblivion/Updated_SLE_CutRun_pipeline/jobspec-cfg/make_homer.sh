#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cm
#SBATCH --time=10:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=homer
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rj931@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --array=1-6


# THIS IS A TEST TO USE THE IgG AS THE CONTROL INSTEAD OF THE CONTROL. I WILL DO THIS TWICE ALSO
# ONCE FOR KNOCKDOWN WITH IGG AND THEN WITH CONTROL AND IGG

cd store_normal_bam_files

#rm kd_bam.txt ctr_bam.txt bg_region.txt kd_ctr_bg_bam.txt


ls 454*.bam > kd_bam.txt
ls 455*.bam > ctr_bam.txt
ls ../bg_sort_bam_files/*.bam > bg_region.txt
paste kd_bam.txt ctr_bam.txt bg_region.txt > kd_ctr_bg_bam.txt

path_to_bg='../bg_sort_bam_files/'

pair_bams='kd_ctr_bg_bam.txt'

line="$(less "$pair_bams" | head -n ${SLURM_ARRAY_TASK_ID} | tail -n 1)"
bam_kd="$(printf "%s" "${line}"| cut -f1)"
bam_ctr="$(printf "%s" "${line}"| cut -f2)"
bam_bg="$(printf "%s" "${line}"| cut -f3)"

module load homer/4.11
module load samtools/intel/1.14

makeTagDirectory $bam_kd'_tag_dir/' $bam_kd -tbp 1
makeTagDirectory $bam_ctr'_tag_dir/' $bam_ctr -tbp 1
makeTagDirectory $bam_bg'_tag_dir/' $path_to_bg$bam_bg -tbp 1


# here are the changes. targets are kd and ctr tag directories, while in both the input is the IgG tag directories
findPeaks $bam_kd'_tag_dir/' -style factor -i $bam_bg'_tag_dir/' -o $bam_kd'_peaks.txt'

findPeaks $bam_ctr'_tag_dir/' -style factor -i $bam_bg'_tag_dir/' -o $bam_ctr'_peaks.txt'



findPeaks $bam_bg'_tag_dir/' -style factor -o $bam_bg'_bg_peaks.txt'



# need to change the chromosome number from just number to chr number

#cat $bam_kd'_peaks.txt' | awk -F '\t' -vOFS='\t' '{$2 = "chr" $2 }1' > $bam_kd'_chr_peaks.txt'

#cat $bam_ctr'_peaks.txt' | awk -F '\t' -vOFS='\t' '{$2 = "chr" $2 }1' > $bam_ctr'_chr_peaks.txt'

#cat $bam_bg'_bg_peaks.txt' | awk -F '\t' -vOFS='\t' '{$2 = "chr" $2 }1' > $bam_bg'_chr_bg_peaks.txt'


#now i want to convert peaks.txt files into bed files, so i can use them in R CHIC package

pos2bed.pl $bam_kd'_peaks.txt' > $bam_kd'_peakfile.bed'

# added this one line for the conversion of the new control peak to bed

pos2bed.pl $bam_ctr'_peaks.txt' > $bam_ctr'_peakfile.bed'

pos2bed.pl $bam_bg'_bg_peaks.txt' > $bam_bg'_bg_peakfile.bed'



echo $bam_bg
echo $path_to_bg$bam_bg

#ref="/scratch/work/courses/BI7653/hw3.2023/hg38/ref_genome"
#ref="/scratch/work/courses/BI7653/hw3.2023/hg38/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa" 

# try the same ref genome from the RNA seq pipeline I downloaded from gencode
ref='/scratch/rj931/tf_sle_project/GRCh38.primary_assembly.genome.fa'

#findMotifsGenome.pl $bam_kd'_peaks.txt' $ref $bam_kd'_motifOutput/' -size 200 -bg $bam_bg'_bg_peaks.txt' 

# added this new line to see the motifs in the control also. maybe if the above works, this would be needed
#findMotifsGenome.pl $bam_ctr'_peaks.txt' $ref $bam_ctr'_motifOutput/' -size 200 -bg $bam_bg'_bg_peaks.txt'

# make an annotation file 
# using the peaks.txt files because there is little guess work on if its in the correct format
# note i removed the -annStats annotate.log option because i am not sure if its needed. check advanced 
# annotation options in homer
# also removed the -gtf $gtf_genes option. This resulted in many columns being NA

#gtf_genes='/scratch/rj931/tf_sle_project/hg38.knownGene.gtf.gz'
# use the gtf file i downloaded and parsed from the rna seq pipeline
gtf_genes='/scratch/rj931/tf_sle_project/test_this.gtf'


mkdir annotated_genes_ds

annotatePeaks.pl $bam_kd'_peaks.txt' $ref -gtf $gtf_genes > $bam_kd'_annotated.tsv'

annotatePeaks.pl  $bam_ctr'_peaks.txt' $ref -gtf $gtf_genes > $bam_ctr'_annotated.tsv'

mv $bam_kd'_annotated.tsv' $bam_ctr'_annotated.tsv' annotated_genes_ds 


##############
# test
#############
#annotatePeaks.pl $bam_kd'_chr_peaks.txt' hg38 > $bam_kd'_annotated.tsv'

#annotatePeaks.pl  $bam_ctr'_chr_peaks.txt' hg38 > $bam_ctr'_annotated.tsv'

#mv $bam_kd'_annotated.tsv' $bam_ctr'_annotated.tsv' annotated_genes_ds


#########################################
#mkdir genes_downstream



# now I want to use bedtools to find the genes downstream of the peaks.
# i needed to get a gene gtf file that was created from a sorted gff file taken from ncbi. and unzip it. 
#here is the path to it

#gtf_genes='/scratch/rj931/tf_sle_project/'

# now we load bedtools and continue

module load bedtools/intel/2.29.2

# here i want to get the intersection of peaks that appear in both the knockdown and 
# control peak files
# we do not want to find intersection between kd and ctr anymore
#bedtools intersect -a $bam_kd'_peakfile.bed' -b $bam_ctr'_peakfile.bed' -f 0.50 -r > $bam_kd'_'$bam_ctr'_both.bed'


# sort the kd and ctr bed files separately  

sort -k1,1 -k2,2n $bam_kd'_peakfile.bed' > $bam_kd'_sorted_final.bed'
sort -k1,1 -k2,2n $bam_ctr'_peakfile.bed' > $bam_ctr'_sorted_final.bed'


# then i take that intersection and fine genes that appear immediately downstream of 
# the peaks 

# option -d is to report the distance of b from a as an extra column, overlapping will be 0
# option -io ignore features in B that overlap A. we want close not touching features
# option -iu ignore features in B that are upstream of features in A
# option -t how to determine between a tie. i want all
# the -iu option requires -D option with 'a' as the orientation

# now we find the genes downstream of the peaks in the knockdown and control sorted bed files separately

#bedtools closest -d -io -iu -t all -D a -a $bam_kd'_sorted_final.bed' -b $gtf_genes > \
#'./genes_downstream/'$bam_kd'_.genes.nearest.txt'

#bedtools closest -d -io -iu -t all -D a -a $bam_ctr'_sorted_final.bed' -b $gtf_genes > \
#'./genes_downstream/'$bam_ctr'_.genes.nearest.txt'


###########
#test above 
##########

#bedtools closest -d -io -iu -t all -D a $bam_kd'_sorted_final.bed' -b $gtf_genes > \
#'./genes_downstream/'$bam_kd'_.genes.nearest.txt'

#bedtools closest -d -io -iu -t all -D a -a $bam_ctr'_sorted_final.bed' -b $gtf_genes > \
#'./genes_downstream/'$bam_ctr'_.genes.nearest.txt'
