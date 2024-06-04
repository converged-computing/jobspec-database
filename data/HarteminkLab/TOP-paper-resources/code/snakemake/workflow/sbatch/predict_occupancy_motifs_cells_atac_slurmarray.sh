#!/bin/bash

#SBATCH --job-name=predict_atac
#SBATCH --output=predict_atac_%A_%a.out
#SBATCH --mem=15G
#SBATCH -p scavenger

## Parameters
module load R/4.1.1-rhel8

METADATA='/datacommons/harteminklab/kl124/TOP/data/ENCODE/metadata/predictions/hg38/ATAC_JASPAR2022NR_all_motifs_predict_data_table.tsv'
DATADIR='/hpc/home/kl124/work/TOP/predict_JASPAR2022_202204/atac_sites_counts/hg38/JASPAR2022'
SITESDIR='/hpc/home/kl124/work/TOP/predict_JASPAR2022_202204/candidate_sites/hg38/JASPAR2022'
GENOMECOUNTSDIR='/hpc/home/kl124/work/TOP/predict_JASPAR2022_202204/atac_genome_counts/hg38'
BAMDIR='/hpc/home/kl124/work/data/ENCODE/ATAC-seq/hg38'
OUTDIR='/hpc/home/kl124/datacommons/TOP/predict_JASPAR2022_202204/hg38/ATAC/JASPAR2022/predicted_occupancy'

tf_list_file='/datacommons/harteminklab/kl124/TOP/data/ENCODE/metadata/predictions/hg38/JASPAR2022_CORE_vertebrates_nr_filtered_TF_motif_table.tsv'

echo "File listing TFs and cell types: ${tf_list_file}"
echo "Run #${SLURM_ARRAY_TASK_ID} on the tf list."
idx_line=${SLURM_ARRAY_TASK_ID}
tf_name=$(awk -v i=${idx_line} -v j=1 'FNR == i {print $j}' ${tf_list_file})
pwm_id=$(awk -v i=${idx_line} -v j=2 'FNR == i {print $j}' ${tf_list_file})
echo "${tf_name}, ${pwm_id}"

# if [[ "${pwm_id}" =~ ^('MA1125.1'|'MA1596.1'|'MA1587.1'|'MA1723.1'|'MA1107.2')$ ]]; then
#    echo "Try large memory for motif ${pwm_id}."
#    exit 1
# fi

cd /hpc/group/harteminklab/kl124/projects/TOP/TOP_workflow/snakemake/workflow/scripts/

start=`date +%s`

Rscript predict_occupancy_motifs_cells.R \
  --metadata ${METADATA} \
  --tf_list ${tf_name} \
  --use_model 'ATAC' \
  --data_dir ${DATADIR} \
  --sites_dir ${SITESDIR} \
  --genomecount_dir ${GENOMECOUNTSDIR} \
  --idxstats_dir ${BAMDIR} \
  --outdir ${OUTDIR}

end=`date +%s`

runtime=$((end-start))
echo "Run time: ${runtime} seconds"
echo "Results saved at: ${OUTDIR}"
