#PBS -S /bin/bash
#PBS -q batch
#PBS -N genotype
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=2:00:00
#PBS -l mem=10gb
#PBS -M hcm14449@uga.edu
#PBS -m abe

# Variant effect predictor (VEP)
# https://useast.ensembl.org/info/docs/tools/vep/script/vep_cache.html#gff
# https://wiki.gacrc.uga.edu/wiki/Ensembl-VEP-Sapelo2
# https://useast.ensembl.org/info/docs/tools/vep/script/vep_options.html

HTSlib_module="HTSlib/1.8-foss-2018a"
gff_file="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.est_evidence.gff3"
workDir="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/"
vcf_file="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/H0/H0_noLow_noHigh_redGem_AncCalls_NoHetsVars.vcf"
out_file="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/H0/vep.test.txt"
genome_fasta="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta"

grep -v "#" 337.nuclear_genome.maker.raw.gff3 | sort -k1,1 -k4,4n -k5,5n -t$'\t' | bgzip -c > 337.nuclear_genome.maker.raw.gff3.gz
tabix -p gff /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.maker.raw.gff3.gz
bgzip -c /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta > /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta.gz




# 337.nuclear_genome.maker.raw.gff3
# # need to make a gtf file from gff3 file for use in cache building for VEP
gffread ${workDir}/337.nuclear_genome.maker.raw.gff3 -T -o ${workDir}/337.nuclear_genome.maker.raw.gtf

# building the cache

# module load VEP/95.0-foss-2018b-Perl-5.28.0

# doing this on my computer now
# perl gtf2vep.pl -i /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/337.nuclear_genome.maker.raw.gtf -f /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/genome.337.fasta -d 78 -s s_par_337
#
#
module load HTSlib/1.8-foss-2018a
# grep -v "#" /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.maker.raw.gtf | sort -k1,1 -k4,4n -k5,5n -t$'\t' | bgzip -c > /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.maker.raw.gtf.gz
#
# module load tabix/0.2.6-foss-2016b
# tabix -p gtf /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.maker.raw.gtf.gz
#
# Bio::EnsEMBL::Registry->load_registry_from_db(-host    => 'ensembldb.ensembl.org',-user    => 'anonymous',-verbose => '1');
#


# /Users/hollymcqueary/ensembl-vep/vep --custom /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/337.nuclear_genome.est_evidence.gff3.gz, 337_anno, gff, overlap, 0
#
# vep -i /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/VCFs/H0/H0_snps_final.vcf \
# --offline \
# --gff /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/337.nuclear_genome.est_evidence.gff3.gz \
# --fasta /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/genome.337.fasta

# ########################################################################################################
#
docker run -t -i -v \
/Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/:/opt/vep/.vep ensemblorg/ensembl-vep \
./vep -i /opt/vep/.vep/D0_snps_final.vcf --force_overwrite --tab -o /opt/vep/.vep/D0_SNPs_VEP.txt \
--gff /opt/vep/.vep/337.nuclear_genome.maker.raw.gff3.gz -fasta /opt/vep/.vep/genome.337.fasta.gz

docker run -t -i -v /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/:/opt/vep/.vep ensemblorg/ensembl-vep \
./vep -i /opt/vep/.vep/D0_Indels_final.vcf --force_overwrite --tab -o /opt/vep/.vep/D0_indels_VEP.txt \
--gff /opt/vep/.vep/337.nuclear_genome.maker.raw.gff3.gz -fasta /opt/vep/.vep/genome.337.fasta.gz

#
# grep -v "#" /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.maker.raw.gff3| sort -k1,1 -k4,4n -k5,5n -t$'\t' | bgzip -c > scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.est_evidence.gff3.gz
# tabix -p gff /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.est_evidence.gff3.gz
# ./vep -i input.vcf --gff data.gff.gz --fasta genome.fa.gz
#

grep -v "#" /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/337.nuclear_genome.est_evidence.gff3 | sort -k1,1 -k4,4n -k5,5n -t$'\t' | bgzip -c > /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/337.nuclear_genome.est_evidence.gff3.gz
/Users/hollymcqueary/ensembl-vep/htslib/tabix -p gff /Users/hollymcqueary/Dropbox/McQueary/Paradoxus_MA/Ref_Genome/337.nuclear_genome.est_evidence.gff3.gz


# singularity exec /usr/local/singularity-images/ensembl-vep.simg vep -i /home/hcm14449/Github/TE_MA/H0_snps_final.vcf \
# -gff /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/337.nuclear_genome.maker.raw.gff3.gz \
# --fasta /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta





# perl variant_effect_predictor.pl -offline -i ${vcf_file} -s s_par_337
#
#
#
#
#
# singularity exec /usr/local/singularity-images/ensembl-vep.simg vep -i ${vcf_file} -o ${out_file} --gff ${workDir}/337.nuclear_genome.est_evidence.gff3.gz --fasta /scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta.gz

# singularity exec /usr/local/singularity-images/ensembl-vep.simg/ensembl-vep.simg which vep
#

# singularity exec ./ensembl-vep.simg vep [options]
