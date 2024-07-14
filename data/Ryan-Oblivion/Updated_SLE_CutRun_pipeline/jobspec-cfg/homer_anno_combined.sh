#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cm
#SBATCH --time=10:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=anno_homer
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rj931@nyu.edu
#SBATCH --output=slurm_%j.out


module load homer/4.11

module load singularity-ce/3.11.0

module load r/gcc/4.3.1

#singularity run /scratch/rj931/containers/r_container.sif

#condition_name='mock'
#condition_name='iav'
condition_name='bleo'

kd_tag_dir='store_normal_bam_files/454*bam_tag_dir/'
ctr_tag_dir='store_normal_bam_files/455*bam_tag_dir/'
bg_tag_dir='bg_sort_bam_files/461*bam_tag_dir'

ref='/scratch/rj931/tf_sle_project/GRCh38.primary_assembly.genome.fa'

# try this same file but with a different name so homer can recognize it
gtf_genes='/scratch/rj931/tf_sle_project/test_this.gtf'
#gtf_genes='/scratch/rj931/tf_sle_project/GRCh38.primary_assembly.genome.fa.annotation'

#getDifferentialPeaksReplicates.pl -t $kd_tag_dir \
#-b $ctr_tag_dir \
#-i $bg_tag_dir \
#-genome $ref \
#-style factor \
#-balanced \
#-gtf $gtf_genes \
#-DESeq2 > outputPeaks.txt 

 
#annotatePeaks.pl 'outputPeaks.txt' $ref -gtf $gtf_genes > 'differential_peaks_annotated.tsv'


######
# trying second way
#####

mkdir differential_peak_analysis

getDifferentialPeaksReplicates.pl -t $kd_tag_dir \
-b $bg_tag_dir \
-i $ctr_tag_dir \
-genome $ref \
-style factor \
-q 0.05 \
-gtf $gtf_genes \
-DESeq2 > outputPeaks_2.txt

annotatePeaks.pl 'outputPeaks_2.txt' $ref -gtf $gtf_genes > 'differential_peak_analysis/differential_'$condition_name'_peaks_annotated_kd_vs_ctr.tsv'


# i jsut need to pool together all the target directories and then find motifs
findPeaks $kd_tag_dir -style factor -o $condition_name'_kd_peaks.txt'
findPeaks $ctr_tag_dir -style factor -o $condition_name'_ctr_peaks.txt'
findPeaks $bg_tag_dir -style factor -o $condition_name'_bg_peaks.txt'

findMotifsGenome.pl $condition_name'_kd_peaks.txt' $ref $condition_name'_kd_combined_diff_analysis_motifOutput/' -size 200 -bg $condition_name'_bg_peaks.txt'
findMotifsGenome.pl $condition_name'_ctr_peaks.txt' $ref $condition_name'_ctr_combined_diff_analysis_motifOutput/' -size 200 -bg $condition_name'_bg_peaks.txt'


#findMotifsGenome.pl 'outputPeaks_2.txt' $ref 'combined_diff_analysis_motifOutput/' -size 200 -bg $bam_bg'_bg_peaks.txt' 

# store_normal_bam_files/454*peaks.txt
# store_normal_bam_files/455*peaks.txt

#kd_peaks='store_normal_bam_files/454*peaks.txt'
#ctr_peaks='store_normal_bam_files/455*peaks.txt'

# might not need this. useful for looking at statistics and creating venn diagram
# mergePeaks $kd_peaks > kd_newPeakFile.txt
# mergePeaks $ctr_peaks > ctr_newPeakFile.txt 

# will have to do this step with an array job, not a list of peaks and tag dirs
#getDifferentialPeaks $kd_peaks $kd_tag_dir $bg_tag_dir  > v2_diff_peaks_kd.txt
#getDifferentialPeaks $ctr_peaks $ctr_tag_dir $bg_tag_dir > v2_diff_peaks_kd.txt