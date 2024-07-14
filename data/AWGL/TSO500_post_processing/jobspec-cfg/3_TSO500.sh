#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --output=%j-%N-3_TSO500.out
#SBATCH --error=%j-%N-3_TSO500.err
#SBATCH --partition=high
#SBATCH --cpus-per-task=20

# Description: Combine all sample results and generate database input (must be run at the 
#              end as NTC is required)
# Use:         from /Output/results/<run_id>/TSO500/ directory, run: 
#              sbatch --export=raw_data=/data/raw/novaseq/<run_id> 3_TSO500.sh
# Version:     1.0.13

##############################################################################################
#  Setup
##############################################################################################

# define filepaths for app
app_version=2.2.0
app_dir=/data/diagnostics/pipelines/TSO500/illumina_app/TSO500_RUO_LocalApp-"$app_version"

# define filepaths for post processing - UPDATE BEFORE GOING LIVE!
pipeline_version=main
pipeline_dir=/data/diagnostics/pipelines/TSO500/TSO500_post_processing-"$pipeline_version"
pipeline_scripts="$pipeline_dir"/scripts

# setup analysis folders
cd "$SLURM_SUBMIT_DIR"
mkdir Gathered_Results
mkdir BAMs

# load singularity and anaconda modules
module purge
module load singularity
. ~/.bashrc
module load anaconda

# catch fails early and terminate
set -euo pipefail


# activate conda env
set +u
conda activate TSO500_post_processing
set -u

# make folder for database output
mkdir Gathered_Results/Database

##############################################################################################
#  Generate inputs for variants database - RNA
#  filter fusions by referral type for contamination script
##############################################################################################

# create fusions table in correct format to import to database
for worksheet_id in $(cat worksheets_rna.txt); do

    # pull out NTC read count
    ntc_reads=$(samtools view -F4 -c analysis/NTC-"$worksheet_id"/Logs_Intermediates/RnaMarkDuplicates/NTC-"$worksheet_id"/NTC-"$worksheet_id".bam)

    for line in $(cat samples_correct_order_"$worksheet_id"_RNA.csv); do
        sample="$(echo "$line" | cut -d, -f1)"
        worksheet_id=$(echo "$line" | cut -d, -f2)
        referral=$(echo "$line" | cut -d, -f4)
        sample_reads=$(tail -n1 analysis/"$sample"/"$sample"_RNA_QC.txt | cut -f5)

        # *AllFusions file isn't made if the app doesnt finish, make a blank one instead
        if [[ ! -f ./analysis/"$sample"/Results/"$sample"/"$sample"_AllFusions.csv ]]; then
            echo "fusion,exons,reference_reads_1,reference_reads_2,fusion_supporting_reads,left_breakpoint,right_breakpoint,type,in_ntc,spanning_reads,spanning_reads_dedup,split_reads,split_reads_dedup,fusion_caller,fusion_score" > ./Gathered_Results/Database/"$sample"_fusion_check.csv

        # if app completes properly, format fusions for database upload
        else
            python "$pipeline_scripts"/fusions2db.py \
              --tsvfile ./analysis/"$sample"/Results/"$sample"/"$sample"_CombinedVariantOutput.tsv \
              --ntcfile ./analysis/NTC-"$worksheet_id"/Results/NTC-"$worksheet_id"/NTC-"$worksheet_id"_CombinedVariantOutput.tsv \
              --allfusions ./analysis/"$sample"/Results/"$sample"/"$sample"_AllFusions.csv \
              --outfile ./Gathered_Results/Database/
        fi

        # copy BAMs
        cp analysis/"$sample"/Logs_Intermediates/RnaMarkDuplicates/"$sample"/"$sample".bam BAMs
        cp analysis/"$sample"/Logs_Intermediates/RnaMarkDuplicates/"$sample"/"$sample".bam.bai BAMs
        
        # combined RNA QC files
        if [[ ! -f RNA_QC_combined.txt ]]; then
            cat analysis/"$sample"/"$sample"_RNA_QC.txt > RNA_QC_combined.txt
        else
            cat analysis/"$sample"/"$sample"_RNA_QC.txt | tail -n1 >> RNA_QC_combined.txt
        fi

        # make database upload sample list
        if [[ "$sample" != NTC* ]]; then
            echo "$sample","$worksheet_id",RNA,"$referral","$sample_reads","$ntc_reads" >> Gathered_Results/Database/samples_database_"$worksheet_id"_RNA.csv
        fi
	done
done


##############################################################################################
#  QC
##############################################################################################

# run contamination script for RNA
for worksheet_id in $(cat worksheets_rna.txt); do
    python "$pipeline_scripts"/contamination_TSO500.py "$worksheet_id" "$pipeline_version"
done

# move sample log files into their own folders
cat samples_correct_order_*_RNA.csv | while read line
do
    sample=$(echo ${line} | cut -f 1 -d ",")
    mv "$sample"_2_TSO500*.out analysis/"$sample"
    mv "$sample"_2_TSO500*.err analysis/"$sample"
done

# add timings
now=$(date +"%T")
echo "End time:   $now" >> timings.txt

# deactivate env
set +u
conda deactivate
set -u

# add marker for run complete
touch run_complete.txt
