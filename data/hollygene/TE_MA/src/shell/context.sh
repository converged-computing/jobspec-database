#PBS -S /bin/bash
#PBS -q batch
#PBS -N getfasta
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=12:00:00
#PBS -l mem=20gb
#PBS -M hcm14449@uga.edu
#PBS -m abe


# script for finding nucleotide context of SNPs

# using bedops to go from vcf to bed file
# then using getfasta to get nucleotide context
awk 'NR==FNR{a[$1,$2]; next} (($1,$2) in a)' SNMs_H0_daveCalls.txt H0_FullCohort_Unfiltered.vcf > H0_snps_final.vcf
awk 'NR==FNR{a[$1,$2]; next} (($1,$2) in a)' indels_H0_daveCalls.txt H0_FullCohort_Unfiltered.vcf > H0_indels_final.vcf


awk 'NR==FNR{a[$1,$2]; next} (($1,$2) in a)' snp_locations_D0.txt D0_FullCohort_Unfiltered.vcf > D0_snps_final.vcf
awk 'NR==FNR{a[$1,$2]; next} (($1,$2) in a)' Indels_D0.txt D0_FullCohort_Unfiltered.vcf > D0_Indels_final.vcf

awk 'NR==FNR{a[$1,$2]; next} (($1,$2) in a)' snpPosD1.txt D1_FullCohort_Unfiltered.vcf > D1_snps_final.vcf
awk 'NR==FNR{a[$1,$2]; next} (($1,$2) in a)' indels_noTy1_D1.txt D1_FullCohort_Unfiltered.vcf > D1_Indels_final.vcf


module load BEDOPS/2.4.30
time vcf2bed < /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/H0/H0_snps_final.vcf > /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/H0/H0_snps_final_HM.bed

time vcf2bed < /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D0/D0_snps_final.vcf > /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D0/D0_snps_final_HM.bed
time vcf2bed < /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D1/D1_snps_final.vcf > /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D1/D1_snps_final_HM.bed


## use awk to subtract 1 from first position and add 1 to second position
# this will give you the trinucleotide context

awk '{$2-=1;$3+=1}1' OFS='\t' H0_snps_final.bed > H0_snps_final_HM_tri.bed
awk '{$2-=1;$3+=1}1' OFS='\t' D0_snps_final_HM.bed > D0_snps_final_HM_tri.bed
awk '{$2-=1;$3+=1}1' OFS='\t' D1_snps_final_HM.bed > D1_snps_final_HM_tri.bed

## use bedtools getfasta to get trinucleotide context

module load BEDTools/2.29.2-GCC-8.2.0-2.31.1

bedtools getfasta -tab -fi /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta -bed /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D0/D0_snps_final_HM_tri.bed -fo /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D0/D0_snps_context.txt

bedtools getfasta -tab -fi /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta -bed /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D1/D1_snps_final_HM_tri.bed -fo /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D1/D1_snps_context.txt
