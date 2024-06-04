#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=8gb
#SBATCH --array=1-24

INDIR=/gpfs/group/pipkin/hdiao/Exp337/0_fastq
BAMDIR=/gpfs/group/pipkin/hdiao/Exp337/1_bowtie2
cd $INDIR

##### Load modules
module load trimgalore
module load bowtie2
module load samtools
module load bedtools
module load python
module load ucsc_tools
module load bamtools
# Split strand reference: https://broadinstitute.github.io/picard/explain-flags.html

INDIR=/gpfs/group/pipkin/hdiao/Exp337/0_fastq
BAMDIR=/gpfs/group/pipkin/hdiao/Exp337/1_bowtie2

##### Last step: failed to load bamtools

##### Split strand
#---- Names
bam_name_srt_flt=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt.bam

bam_name_srt_flt_Rm=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_Rm.bam
bam_name_srt_flt_RmMm=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_RmMm.bam
bam_name_srt_flt_RmMm_cor=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_RmMm_cor.bam

bam_name_srt_flt_r1=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_r1.bam
bam_name_srt_flt_r2=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_r2.bam

bam_name_srt_flt_r1_F=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_r1_F.bam
bam_name_srt_flt_r1_R=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_r1_R.bam
bam_name_srt_flt_r2_F=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_r2_F.bam
bam_name_srt_flt_r2_R=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_r2_R.bam

bam_name_srt_flt_pos=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_pos.bam
bam_name_srt_flt_neg=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_neg.bam
bam_name_srt_flt_pos_srt=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_pos_srt.bam
bam_name_srt_flt_neg_srt=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_neg_srt.bam

bdg_name_srt_flt_pos=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_pos.bdg
bdg_name_srt_flt_neg=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_neg.bdg

bdg_name_srt_flt_pos_chr=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_pos_chr.bdg
bdg_name_srt_flt_neg_chr=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_neg_chr.bdg

bdg_name_srt_flt_pos_chr_srt=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_pos_chr_srt.bdg
bdg_name_srt_flt_neg_chr_srt=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_srt_flt_neg_chr_srt.bdg

bw_name_srt_flt_pos_chr_srt=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_pos.bw
bw_name_srt_flt_neg_chr_srt=$BAMDIR/337_${SLURM_ARRAY_TASK_ID}_neg.bw

#---- Filter r1 / r2
# Remove reads that are unmapped
# Remove reads whose mates are unmapped
# Keep only reads that are properly paired
#samtools view -F 0x4 -h -b $bam_name_srt_flt > $bam_name_srt_flt_Rm 
#samtools view -F 0x8 -h -b $bam_name_srt_flt_Rm > $bam_name_srt_flt_RmMm 
#samtools view -f 0x2 -h -b $bam_name_srt_flt_RmMm > $bam_name_srt_flt_RmMm_cor 

# Filter out r1 / r2 seperatly
#samtools view -f 0x40 -h -b $bam_name_srt_flt_RmMm_cor > $bam_name_srt_flt_r1
#samtools view -f 0x80 -h -b $bam_name_srt_flt_RmMm_cor > $bam_name_srt_flt_r2

# Filter out r1 that are positive / r2 that are negative
#samtools view -f 0x20 -h -b $bam_name_srt_flt_r1 > $bam_name_srt_flt_r1_F
#samtools view -f 0x10 -h -b $bam_name_srt_flt_r2 > $bam_name_srt_flt_r2_R
bamtools merge -in $bam_name_srt_flt_r1_F -in $bam_name_srt_flt_r2_R -out $bam_name_srt_flt_pos

# Filter out r1 that are negative / r2 that are positive
samtools view -f 0x10 -h -b $bam_name_srt_flt_r1 > $bam_name_srt_flt_r1_R
samtools view -f 0x20 -h -b $bam_name_srt_flt_r2 > $bam_name_srt_flt_r2_F
bamtools merge -in $bam_name_srt_flt_r1_R -in $bam_name_srt_flt_r2_F -out $bam_name_srt_flt_neg


samtools sort $bam_name_srt_flt_pos -o $bam_name_srt_flt_pos_srt
samtools sort $bam_name_srt_flt_neg -o $bam_name_srt_flt_neg_srt
samtools index $bam_name_srt_flt_pos_srt
samtools index $bam_name_srt_flt_neg_srt

#---- Bam to Bdg
bamCoverage --bam $bam_name_srt_flt_pos_srt -o $bdg_name_srt_flt_pos  --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2652783500 --extendReads --outFileFormat bedgraph
bamCoverage --bam $bam_name_srt_flt_neg_srt -o $bdg_name_srt_flt_neg  --binSize 10 --normalizeUsing RPGC --effectiveGenomeSize 2652783500 --extendReads --outFileFormat bedgraph

#---- Bdg add chr
awk '{print "chr"$1 "\t" $2 "\t" $3 "\t" $4}' $bdg_name_srt_flt_pos > $bdg_name_srt_flt_pos_chr
awk '{print "chr"$1 "\t" $2 "\t" $3 "\t" $4}' $bdg_name_srt_flt_neg > $bdg_name_srt_flt_neg_chr

#---- Sort bdg
LC_COLLATE=C sort -k1,1 -k2,2n $bdg_name_srt_flt_pos_chr > $bdg_name_srt_flt_pos_chr_srt
LC_COLLATE=C sort -k1,1 -k2,2n $bdg_name_srt_flt_neg_chr > $bdg_name_srt_flt_neg_chr_srt

#---- Bdg to bw
chrom_sizes=/gpfs/group/pipkin/hdiao/ref_resources/mm/release102/GRCm38.genome.sizes.withChr
bedGraphToBigWig $bdg_name_srt_flt_pos_chr_srt $chrom_sizes $bw_name_srt_flt_pos_chr_srt
bedGraphToBigWig $bdg_name_srt_flt_neg_chr_srt $chrom_sizes $bw_name_srt_flt_neg_chr_srt

