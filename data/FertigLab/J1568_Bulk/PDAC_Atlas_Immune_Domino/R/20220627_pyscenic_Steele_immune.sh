#!/bin/bash
#SBATCH --job-name=hg38_Steele_immune_TNFRSF9_noNA
#SBATCH --time=24:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=10
#SBATCH --mem=160G
#SBATCH --mail-type=end
#SBATCH --mail-user=jmitch81@jhmi.edu

# directory for storing results
RESULT_DIR="results/20220627_Steele_immune_NA_excluded_hg38"
COUNTS_LOOM="PDAC_immune_Steele/T_immune_subset_seurat_NA_excluded_counts_matrix.loom"
JOB_SCRIPT="scripts/20220627_pyscenic_Steele_immune.sh"
R_SCRIPT="scripts/20220627_domino_Steele_immune.R"

mkdir $RESULT_DIR

# copy the job script to the results directory
cp $JOB_SCRIPT "${RESULT_DIR}/."
cp $R_SCRIPT "${RESULT_DIR}/."

singularity exec ~/venv/domino/aertslab-pyscenic-0.11.0.sif pyscenic grn \
	$COUNTS_LOOM \
	reference/allTFs_hg38.txt \
	-o "${RESULT_DIR}/adj.tsv" \
	--num_workers 10

echo "Step one complete"

singularity exec ~/venv/domino/aertslab-pyscenic-0.11.0.sif pyscenic ctx \
	"${RESULT_DIR}/adj.tsv" \
	reference/HG38/hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.feather \
	reference/HG38/hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.feather \
	--annotations_fname reference/HG38/motifs-v9-nr.hgnc-m0.001-o0.0.tbl \
	--expression_mtx_fname "${COUNTS_LOOM}" \
	--mode "dask_multiprocessing" \
	--output "${RESULT_DIR}/regulons.csv" \
	--num_workers 10

echo "Step two complete"

singularity exec ~/venv/domino/aertslab-pyscenic-0.11.0.sif pyscenic aucell \
	"${COUNTS_LOOM}" \
	"${RESULT_DIR}/regulons.csv" \
	-o "${RESULT_DIR}/auc.csv"

echo "Step three complete"

echo "Begining domino script"

# Execute R script to create the domino object
module load seurat/4.1.1
module list

R CMD BATCH $R_SCRIPT

echo "domino creation complete"

#### execute code and write output file to OUT-24log.
# time mpiexec ./code-mvapich.x > OUT-24log
echo "Finished with job $SLURM_JOBID"
