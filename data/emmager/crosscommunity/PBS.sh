#!/bin/bash

#SBATCH --job-name=Fst
#SBATCH --output=FstChr.%a.out
#SBATCH --error=FstChr.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=8:00:00
#SBATCH --array=1
#SBATCH --mail-type=end
#SBATCH --mail-user=emmarg@princeton.edu


#module load plink

##RUN IN GEN-COMP2
CHROM=${SLURM_ARRAY_TASK_ID}

grep -E "ASW|YRI" 1000G_Phase3_sample_map > ASW_YRI_sample_map
awk '{print "0",$1,$2,$3,$4}' ASW_YRI_sample_map > tmp 
mv tmp ASW_YRI_sample_map
awk '{print $2}' ASW_YRI_sample_map > ASW_YRI_sample_IDs

grep -E "JPT|YRI" 1000G_Phase3_sample_map > JPT_YRI_sample_map
awk '{print "0",$1,$2,$3,$4}' JPT_YRI_sample_map > tmp 
mv tmp JPT_YRI_sample_map
awk '{print $2}' JPT_YRI_sample_map > JPT_YRI_sample_IDs

grep -E "ASW|JPT" 1000G_Phase3_sample_map > ASW_JPT_sample_map
awk '{print "0",$1,$2,$3,$4}' ASW_JPT_sample_map > tmp 
mv tmp ASW_JPT_sample_map
awk '{print $2}' ASW_JPT_sample_map > ASW_JPT_sample_IDs 

grep -E "ASW|JPT|YRI" 1000G_Phase3_sample_map > PBS_sample_map
awk '{print "0",$1,$2,$3,$4}' PBS_sample_map > tmp 
mv tmp PBS_sample_map

plink1=/Genomics/grid/users/alea/programs/plink_1.90
plink2=/Genomics/ayroleslab2/emma/Turkana_Genotyping/1000G_data/bin/plink2

/Genomics/argo/users/emmarg/lab/Turkana_Genotyping/1000G_data/bin/plink2 --bfile ALLCHR.phase3_v5.shapeit2_mvncall_integrated_int_allFilters_subset --make-bed --out forFst --snps-only --geno 0.25 --maf 0.01  --indep-pairwise 50 20 0.8 --within JPT_YRI_sample_map


$plink2 --bfile forFst --extract forFst.prune.in --within ASW_YRI_sample_map POP --fst POP method=hudson report-variants --out ASW_YRI_FST_hud
$plink2 --bfile forFst --extract forFst.prune.in --within JPT_YRI_sample_map POP --fst POP method=hudson report-variants --out JPT_YRI_FST_hud
$plink2 --bfile forFst --extract forFst.prune.in --within ASW_JPT_sample_map POP --fst POP method=hudson report-variants --out ASW_JPT_FST_hud
$plink1 --bfile forFst --out PBS_hud --extract forFst.prune.in --within PBS_sample_map --freq

awk '{OFS="\t"; print $2,$3,$4,$5,$6,$8}' PBS_hud.frq.strat  > PBS_hud.frq_v2.strat
