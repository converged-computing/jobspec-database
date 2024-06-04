### The phased data is not used in the manuscript analyses (updated on 05/2024) ###


#Create and ind list of the used vcf:
cd /scratch/project_2001443/barriers_introgr_formica/vcf/filt
bcftools query -l all_samples.DP8.hwe.AN10.noScaff00.mac2.vcf.gz > all_samples.DP8.hwe.AN10.noScaff00.mac2.ind.list
# contains 101 samples; all except 110 and 105 (even does contain 54, Pus2, s353 and s354)

---

#!/bin/bash -l
#SBATCH -J whap
#SBATCH -o /scratch/project_2001443/barriers_introgr_formica/vcf/phasing/whatshap/logs/whatshap_%a.out
#SBATCH -e /scratch/project_2001443/barriers_introgr_formica/vcf/phasing/whatshap/logs/whatshap_%a.err
#SBATCH --account=project_2001443
#SBATCH -t 24:00:00
#SBATCH -p small
#SBATCH --array=1-101
#SBATCH --ntasks 1
#SBATCH --mem=25G
#SBATCH --mail-type=END

# modules
module load biokit
export PATH="/projappl/project_2001443/whatshapenv/bin:$PATH" 

cd /scratch/project_2001443/barriers_introgr_formica/vcf/phasing

# Define paths
BAMDIR=/scratch/project_2001443/barriers_introgr_formica/bam_all
masterVCF=/scratch/project_2001443/barriers_introgr_formica/vcf/filt/all_samples.DP8.hwe.AN10.noScaff00.mac2.vcf.gz
REF=/scratch/project_2001443/reference_genome/Formica_hybrid_v1_wFhyb_Sapis.fa
INDLIST=/scratch/project_2001443/barriers_introgr_formica/vcf/filt/all_samples.DP8.hwe.AN10.noScaff00.mac2.ind.list

# Define focal ind
ind=$(sed -n "$SLURM_ARRAY_TASK_ID"p $INDLIST)
indbam=${ind%%-*} #in the bam file names, inds sequenced earlier have only numbers as ids, whereas the vcf ids (and INDLIST) contain species info for these. Match these.

# 1. Extract single individual from VCF & fix floats in GQ field (or issue later when parsing with whatshap)
bcftools view -s $ind $masterVCF -Ov | perl -npe 's/(.\/.)\:.+\:(.+\:.+\:.+\:.+\:.+\:.+\:.+)/$1\:99\:$2/' | grep -v '^Scaffold00' | bgzip > whatshap/${ind}.GQfixed.vcf.gz &&\
bcftools index -t whatshap/${ind}.GQfixed.vcf.gz

# 2. Phase
whatshap phase -o whatshap/${ind}.phased.vcf.gz --reference $REF \
               whatshap/${ind}.GQfixed.vcf.gz $BAMDIR/${indbam}_nodupl_*.bam &&\
whatshap stats whatshap/${ind}.phased.vcf.gz --tsv=whatshap/${ind}.phased.tsv

### end
2_whatshap.sh (END)
