#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --output=%j-%N-2_TSO500.out
#SBATCH --error=%j-%N-2_TSO500.err
#SBATCH --partition=high
#SBATCH --cpus-per-task=24

# Description: Run Illumina TSO500 app for each sample then run postprocessing steps - FastQC, 
#              GATK depth of coverage, coverage calculator, bed2hgvs, gather QC metrics. Kick
#              off script 3 when all samples completed
# Use:         from /Output/results/<run_id>/TSO500/ directory, for each sample run: 
#              sbatch --export=raw_data=/data/raw/novaseq/<run_id>,sample_id=<sample_id> 2_TSO500.sh
# Version:     1.0.13

##############################################################################################
#  Setup
##############################################################################################

# define filepaths for app
app_version=2.2.0
app_dir=/data/diagnostics/pipelines/TSO500/illumina_app/TSO500_RUO_LocalApp-"$app_version"

# define filepaths for post processing
pipeline_version=main
pipeline_dir=/data/diagnostics/pipelines/TSO500/TSO500_post_processing-"$pipeline_version"
pipeline_scripts="$pipeline_dir"/scripts

# setup analysis folders
cd "$SLURM_SUBMIT_DIR"
output_path="$SLURM_SUBMIT_DIR"/analysis/"$sample_id"
mkdir -p $output_path

# load singularity and anaconda modules
module purge
module load singularity
. ~/.bashrc
module load anaconda

# catch fails early and terminate
set -euo pipefail

# define pipeline variables
minimum_coverage="270 135"
coverage_bed_files_path="$pipeline_dir"/hotspot_coverage
vendor_capture_bed="$pipeline_dir"/vendorCaptureBed_100pad_updated.bed
preferred_transcripts="$pipeline_dir"/preferred_transcripts.txt
worksheet=$(grep "$sample_id" SampleSheet_updated.csv | cut -d, -f3)
dna_or_rna=$(grep "$sample_id" SampleSheet_updated.csv | cut -d, -f8)


##############################################################################################
#  Illumina app
##############################################################################################

# use sampleOrPairIDs flag to run one sample at a time
"$app_dir"/TruSight_Oncology_500_RUO.sh \
  --analysisFolder "$output_path" \
  --resourcesFolder "$app_dir"/resources \
  --fastqFolder "$SLURM_SUBMIT_DIR"/Demultiplex_Output/Logs_Intermediates/FastqGeneration \
  --isNovaSeq \
  --sampleSheet "$raw_data"/SampleSheet.csv \
  --engine singularity \
  --sampleOrPairIDs "$sample_id"


##############################################################################################
#  FastQC
##############################################################################################

# activate conda env
set +u
conda activate TSO500_post_processing
set -u

# make fastqc output folder in the sample folder
fastqc_output="$output_path"/FastQC/
mkdir -p $fastqc_output

# location of sample fastqs
fastq_path="$SLURM_SUBMIT_DIR"/Demultiplex_Output/Logs_Intermediates/FastqGeneration/"$sample_id"/

# run FastQC for each fastq pair
for fastqPair in $(find $fastq_path -name *.fastq.gz -type f -printf "%f\n" | cut -d_ -f1-3 | sort | uniq); do

    #parse fastq filenames
    laneId=$(echo basename "$fastqPair" | cut -d_ -f3)
    read1Fastq=$(ls "$fastq_path""$fastqPair"_R1_*fastq.gz)
    read2Fastq=$(ls "$fastq_path""$fastqPair"_R2_*fastq.gz)

    # run FastQC
    fastqc --extract -o "$fastqc_output" "$read1Fastq"
    fastqc --extract -o "$fastqc_output" "$read2Fastq"

    # rename files
    mv "$fastqc_output"/"$sample_id"_S*_"$laneId"_R1_001_fastqc/summary.txt "$fastqc_output"/"$sample_id"_"$laneId"_R1_fastqc.txt
    mv "$fastqc_output"/"$sample_id"_S*_"$laneId"_R2_001_fastqc/summary.txt "$fastqc_output"/"$sample_id"_"$laneId"_R2_fastqc.txt

done


##############################################################################################
#  DNA only steps
##############################################################################################

if [ "$dna_or_rna" = "DNA" ]; then


    #-------------------------------------------------------------------------------------
    #  Call variants outside of app ROI
    #-------------------------------------------------------------------------------------

    bash "$pipeline_scripts"/call_extra_padding_variants.sh "$sample_id" "$pipeline_version"


    #-------------------------------------------------------------------------------------
    #  Run depth of coverage with limited bed (whole 500 takes a long time)
    #-------------------------------------------------------------------------------------

    # set and make filepaths for depth of coverage
    bam_path="$output_path"/Logs_Intermediates/StitchedRealigned/"$sample_id"/
    depth_path="$output_path"/depth_of_coverage
    mkdir -p $depth_path

    # reheader the bams to local area
    java -jar /Apps/wren/picard/2.21.6/bin/picard.jar AddOrReplaceReadGroups \
      I="$bam_path"/"$sample_id".bam \
      O="$bam_path"/"$sample_id"_add_rg.bam \
      RGID=4 \
      RGLB=lib1 \
      RGPL=ILLUMINA \
      RGPU=unit1 \
      RGSM=20

    # index new bam
    samtools index "$bam_path"/"$sample_id"_add_rg.bam "$bam_path"/"$sample_id"_add_rg.bam.bai

    # run depth of coverage
    gatk DepthOfCoverage \
      -I "$bam_path"/"$sample_id"_add_rg.bam \
      -L "$vendor_capture_bed" \
      -R "$app_dir"/resources/genomes/hg19_hardPAR/genome.fa \
      -O "$depth_path"/"$sample_id"_depth_of_coverage

    # change to tab delimited and remove colon from column 1
    sed 's/:/\t/g' "$depth_path"/"$sample_id"_depth_of_coverage \
      | sed 's/,/\t/g' | grep -v 'Locus' \
      | sort -k1,1 -k2,2n | bgzip \
      > "$depth_path"/"$sample_id"_depth_of_coverage.gz

    # tabix index depth of coverage file
    tabix \
      -b 2 \
      -e 2 \
      -s 1 \
      "$depth_path"/"$sample_id"_depth_of_coverage.gz

    # deactivate env
    set +u
    conda deactivate
    set -u


    #-------------------------------------------------------------------------------------
    #  Run coverage calculator at 135X and 270X depth cutoffs
    #-------------------------------------------------------------------------------------

    # repeat for each coverage value
    for min_coverage in $minimum_coverage; do

        # activate coverage calculator conda env
        set +u
        conda activate CoverageCalculatorPy
        set -u

        # set output directory for coverage files
        hscov_outdir=hotspot_coverage_"$min_coverage"x

        # run coverage calculator on each bed file
        for bed_file in "$coverage_bed_files_path"/*.bed; do

            name=$(echo $(basename $bed_file) | cut -d"." -f1)

            python /data/diagnostics/apps/CoverageCalculatorPy/CoverageCalculatorPy-v1.1.0/CoverageCalculatorPy.py \
              -B "$coverage_bed_files_path"/"$name".bed \
              -D "$depth_path"/"$sample_id"_depth_of_coverage.gz \
              --depth "$min_coverage" \
              --padding 0 \
              --groupfile "$coverage_bed_files_path"/"$name".groups \
              --outname "$sample_id"_"$name" \
              --outdir  "$depth_path"/"$hscov_outdir"/

            # remove header from gaps file
            if [[ $(wc -l < "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".gaps) -eq 1 ]]; then
                # no gaps
                touch "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".nohead.gaps

            else
                # gaps
                grep -v '^#' "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".gaps > "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".nohead.gaps

            fi

            # remove chr from bed file so bed2hgvs works
            cat "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".nohead.gaps | sed 's/^chr//' > "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".nohead_nochr.gaps

            # remove intermediate files
            rm "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".gaps
            rm "$depth_path"/"$hscov_outdir"/"$sample_id"_"$name".nohead.gaps

        done


        #-------------------------------------------------------------------------------------
        #  Run bed2hgvs to add hgvs nomenclature to gaps
        #-------------------------------------------------------------------------------------
    
        # activate bed2hgvs conda env
        set +u
        conda deactivate
        conda activate bed2hgvs
        set -u

        # run on each bed file
        for gaps_file in "$depth_path"/"$hscov_outdir"/*.nohead_nochr.gaps; do

            name=$(echo $(basename $gaps_file) | cut -d"." -f1)
            echo $name

            Rscript /data/diagnostics/apps/bed2hgvs/bed2hgvs-v0.3.0/bed2hgvs.R \
              --bedfile $gaps_file \
              --outname "$name".gaps \
              --outdir "$depth_path"/"$hscov_outdir" \
              --preferred_tx $preferred_transcripts

            # remove intermediate file
            rm "$depth_path"/"$hscov_outdir"/"$name".nohead_nochr.gaps
        done

        # combine all total coverage files
        if [ -f "$depth_path"/"$hscov_outdir"/"$sample_id"_coverage.txt ]; then rm "$depth_path"/"$hscov_outdir"/"$sample_id"_coverage.txt; fi
        cat "$depth_path"/"$hscov_outdir"/*.totalCoverage | grep "FEATURE" | head -n 1 >> "$depth_path"/"$hscov_outdir"/"$sample_id"_coverage.txt
        cat "$depth_path"/"$hscov_outdir"/*.totalCoverage | grep -v "FEATURE" | grep -vP "combined_\\S+_GENE" >> "$depth_path"/"$hscov_outdir"/"$sample_id"_coverage.txt

        # deactivate env
        set +u
        conda deactivate
        set -u


        #-------------------------------------------------------------------------------------
        #  Cosmic gaps
        #-------------------------------------------------------------------------------------

        # activate conda env
        set +u
        conda activate TSO500_post_processing
        set -u

        cosmic_tool_path=/data/diagnostics/apps/cosmic_gaps/cosmic_gaps-master

        # parse referral - must be in DNA loop
        referral=$(grep "$sample_id" samples_correct_order_"$worksheet"_DNA.csv | cut -d, -f4)
        gaps_file="$depth_path"/"$hscov_outdir"/"$sample_id"_"$referral"_hotspots.gaps

        # hotspot gaps file may be missing for some referrals
        if [[ -f $gaps_file ]]
        then

            # only run bedtools intersect for certain referral types
            if [ $referral = "Melanoma" ] ||  [ $referral = "Lung" ] || [ $referral = "Colorectal" ] || [ $referral = "GIST" ] || [ $referral = "breast" ]
            then
                dos2unix $gaps_file

                # find the overlap between the hotspots file and the referral file from cosmic
                bedtools intersect \
                  -loj \
                  -F 1 \
                  -a $gaps_file \
                  -b "$cosmic_tool_path"/cosmic_bedfiles/"$referral".bed \
                  -wao \
                > "$depth_path"/"$hscov_outdir"/"$sample_id"_"$referral"_intersect.txt

            fi

            # filter the output 
            python "$cosmic_tool_path"/filter_table.py \
              --sampleId $sample_id \
              --referral $referral \
              --gaps_path "$depth_path"/"$hscov_outdir"/ \
              --bedfile_path "$cosmic_tool_path"/cosmic_bedfiles/
        
        fi
    
        # deactivate env
        set +u
        conda deactivate
        set -u

    done

fi



##############################################################################################
#  Gather QC metrics for sample
##############################################################################################

# activate conda env
set +u
conda activate TSO500_post_processing
set -u

# function to check FASTQC output
count_qc_fails() {
    #count how many core FASTQC tests failed
    grep -E "Basic Statistics|Per base sequence quality|Per tile sequence quality|Per sequence quality scores|Per base N content" "$1" | \
    grep -v ^PASS | \
    grep -v ^WARN | \
    wc -l | \
    sed 's/^[[:space:]]*//g'
}

# check all fastqc files for any fails
fastqc_status=PASS

for report in "$fastqc_output"/"$sample_id"_*_fastqc.txt; do
    if [ $(count_qc_fails $report) -gt 0 ]; then
        fastqc_status=FAIL
    fi
done

# pull out metrics from the Illumina app MetricsOutput.tsv
completed_all_steps=$(grep COMPLETED_ALL_STEPS analysis/"$sample_id"/Results/MetricsOutput.tsv | cut -f2)

# DNA only metrics
if [ "$dna_or_rna" = "DNA" ]; then

    contamination_score=$(grep CONTAMINATION_SCORE analysis/"$sample_id"/Results/MetricsOutput.tsv | cut -f4)
    contamination_p_value=$(grep CONTAMINATION_P_VALUE analysis/"$sample_id"/Results/MetricsOutput.tsv | cut -f4)
    total_pf_reads=$(grep TOTAL_PF_READS analysis/"$sample_id"/Results/MetricsOutput.tsv | head -n1 | cut -f4)
    median_insert_size=$(grep MEDIAN_INSERT_SIZE analysis/"$sample_id"/Results/MetricsOutput.tsv | head -n1 | cut -f4)
    median_exon_coverage=$(grep MEDIAN_EXON_COVERAGE analysis/"$sample_id"/Results/MetricsOutput.tsv | cut -f4)
    pct_exon_50x=$(grep PCT_EXON_50X analysis/"$sample_id"/Results/MetricsOutput.tsv | cut -f4)

    contamination_pass_fail=PASS
    if [ $contamination_score -gt 3106 ]; then

        # use python to handle float
        p_value_cutoff=$(python -c "print(float("$contamination_p_value") > 0.049)")
        if [ $p_value_cutoff = "True" ]; then
            contamination_pass_fail=FAIL
        fi
    fi


    #run samtools to get the number of reads in the bam file
    if [[ -f ./analysis/"$sample_id"/Logs_Intermediates/StitchedRealigned/"$sample_id"/"$sample_id".bam ]]; then
        reads=$( samtools view -c ./analysis/"$sample_id"/Logs_Intermediates/StitchedRealigned/"$sample_id"/"$sample_id".bam )
    else
        reads="NA"
    fi

    # add to sample QC file
    echo -e "Sample\tFastQC\tcompleted_all_steps\tcontamination_pass_fail\tcontamination_score\tcontamination_p_value\ttotal_pf_reads\tmedian_insert_size\tmedian_exon_coverage\tpct_exon_50x\tAligned_reads" > "$output_path"/"$sample_id"_"$dna_or_rna"_QC.txt
    echo -e "$sample_id\t$fastqc_status\t$completed_all_steps\t$contamination_pass_fail\t$contamination_score\t$contamination_p_value\t$total_pf_reads\t$median_insert_size\t$median_exon_coverage\t$pct_exon_50x\t$reads" >> "$output_path"/"$sample_id"_"$dna_or_rna"_QC.txt

fi


# RNA only metrics
if [ "$dna_or_rna" = "RNA" ]; then

    median_cv_gene_500x=$(grep "MEDIAN_CV_GENE_500X" analysis/"$sample_id"/Results/MetricsOutput.tsv | cut -f4)
    total_on_target_reads=$(grep "TOTAL_ON_TARGET_READS" analysis/"$sample_id"/Results/MetricsOutput.tsv | tail -n1 | cut -f4)
    median_insert_size=$(grep "MEDIAN_INSERT_SIZE" analysis/"$sample_id"/Results/MetricsOutput.tsv | tail -n1 | cut -f4)
    total_pf_reads=$(grep "TOTAL_PF_READS" analysis/"$sample_id"/Results/MetricsOutput.tsv | tail -n1 | cut -f4)

    # add to sample QC file
    echo -e "Sample\tFastQC\tcompleted_all_steps\tmedian_cv_gene_500x\ttotal_on_target_reads\tmedian_insert_size\ttotal_pf_reads" > "$output_path"/"$sample_id"_"$dna_or_rna"_QC.txt
    echo -e "$sample_id\t$fastqc_status\t$completed_all_steps\t$median_cv_gene_500x\t$total_on_target_reads\t$median_insert_size\t$total_pf_reads" >> "$output_path"/"$sample_id"_"$dna_or_rna"_QC.txt

fi


##############################################################################################
#  Kick off run level script once all samples have finished
##############################################################################################

# add sample to completed list once finished
echo $sample_id >> completed_samples.txt

# only run once all samples have finished
expected=$(cat samples_correct_order_*_RNA.csv | wc -l)
complete=$(cat completed_samples.txt | wc -l)

# if last sample, kick off script 3
if [ "$complete" -eq "$expected" ]; then
    sbatch --export=raw_data="$raw_data" 3_TSO500.sh
fi

# deactivate env
set +u
conda deactivate
set -u
