#!/bin/bash
#SBATCH --job-name=popInf_master # Job name 
#SBATCH --mem-per-cpu=16000
#SBATCH -o slurm.%j.out                # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err                # STDERR (%j = JobId)
#SBATCH -n 1
#SBATCH -t 48:00:00

# ___USAGE___ #
# Please run this script as follows:
# sbatch snakemake_PopInf_slurm.sh A 
# or
# sbatch snakemake_PopInf_slurm.sh X 

# NOTE #
# If you are not running this shell script on a cluster, remove lines 2-7
# and replace the snakemake command on line 65 with just "snakemake"

###############################################################################
# ATTENTION #
# Please change the following variables so that this script runs based on your
# data.

# 1. Path to the location of the Snakefile and corresponding scripts
SPATH=/full/path/to/PopInf/directory/
# 2. Name of the environment you created
ENV=popInf
# 3. Email you want the notifications to be sent to. If running on a cluster. 
# This is the email address you wish to send slurm logs to.
EMAIL=youremail@email.com
# 4. The path to and file name of the reference panel samples list
POPFILEREF=/full/path/to/Sample_Information/ThousandGenomesSamples_AdmxRm.txt
# 5. The path to and file name of the unknown panel samples list
POPFILEUNK=/full/path/to/unknown_sample/Sample_Information/INFO.txt

###############################################################################
date
cd $SPATH
source activate $ENV


echo "Checking whether the autosomes or the X chromosome are to be analyzed."
echo ""

if [[ $1 = "A" ]]; then
	echo "The autosomes will be analyzed."
	echo ""
elif [[ $1 = "X" ]];
then 
	echo "The X chromosome will be analyzed."
	echo ""
else
	echo "***** Please specify if you wish to analyze autosomes (A) or the X chromosome (X) *****"
	echo ""
	echo "Example: "
	echo "bash snakemake_PopInf_slurm.sh A "
	echo "or "
	echo "bash snakemake_PopInf_slurm.sh X"
	exit 1

fi


echo ""
echo "Snakemake will now start"
snakemake -j 22 --latency-wait 20 --cluster "sbatch -n 8 -t 12:00:00 --mail-type=END,FAIL --mail-user=${EMAIL}"
# If you are not running this shell script on the cluster, simply remove the  
# above snakemake command and replace this with just: snakemake 

echo""
echo "Snakemake has finished running, if analyzing autosomes, merged analysis will now start."


if [[ $1 = "A" ]]; then
	echo "The autosomes will be analyzed."
	echo ""
	echo "The merge list is being created."
	echo ""
	# Create the merge list
	python make_merge_list.py --path autosomes/merge/ --stem _reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune --out merge_list

	echo ""
	
	echo "The individual chromosome files are now being merged."
	echo ""
	# Merge the plink files
	plink --file autosomes/merge/chr1_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune --merge-list merge_list.txt --recode --out autosomes/merge/merge_all_chr_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune
	echo ""
	
	# Edit the 6th column of the ped files
	awk '{{$6 = "1"; print}}' autosomes/merge/merge_all_chr_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.ped > autosomes/merge/merge_all_chr_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_editColumn6.ped
	echo ""

	echo "PCA is now starting for the merged autosome file."
	echo ""

	# Make par file
	python make_par.py --map autosomes/merge/merge_all_chr_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.map --ped autosomes/merge/merge_all_chr_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_editColumn6.ped --ev merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune --par merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune_par

	# Run smartpca
	smartpca -p merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune_par_PCA.par
	echo ""
	echo "PCA is complete"

	# Edit the evec files
	awk '{{if($1 == "\t" ) {{print $2,"\t",$3,"\t",$4,"\t",$5,"\t",$6,"\t",$7,"\t",$8,"\t",$9,"\t",$10,"\t",$11,"\t",$12,"\t",$13,"\t"}} else {{print $1,"\t",$2,"\t",$3,"\t",$4,"\t",$5,"\t",$6,"\t",$7,"\t",$8,"\t",$9,"\t",$10,"\t",$11,"\t",$12,"\t"}}}}' merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune.evec > merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune_Fix.evec
	awk '{gsub(/\.variant2/,""); gsub(/\.variant/,""); print}' merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune_Fix.evec > merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune_Fix2.evec
	echo ""

	echo "The results are now being plotted and an inferred ancestry report is being generated for the merged autosomes."
	echo ""

	# Plot results and get inferred population report
	Rscript pca_inferred_ancestry_report.R merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune_Fix2.evec merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune.eval $POPFILEREF $POPFILEUNK autosomes_inferred_pop_report autosomes_inferred_pop_report


else
	echo "Done!"

fi

date
