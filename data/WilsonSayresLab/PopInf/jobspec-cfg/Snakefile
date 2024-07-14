# This snakemake file will analyze the autosomes and the X chromosome separately in two blocks.
# The first block will be for the autosomes that have been separated by chromosome and the second block will be for the
# X chromosome.

# The user must first provide the directories and file names for the reference and unknown panels in the popInf.config.json
configfile: "popInf.config.json"

# PART 1: AUTOSOMES
if config["Autosomes_Yes_or_No"]=="Y":
	rule all:
		input:
			expand("autosomes/ref_set/chr{chrm}_reference_panel_set_SNPs.recode.vcf", chrm=config["chromosome"]),
			expand("autosomes/unk_set/chr{chrm}_unkown_set_SNPs.recode.vcf", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge.vcf", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing.recode.vcf", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.map", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.ped", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.prune.in", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.prune.out", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.map", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.ped", chrm=config["chromosome"]),
			expand("autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_editColumn6.ped", chrm=config["chromosome"]),
			expand("autosomes/pca/par/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_par_PCA.par", chrm=config["chromosome"]),
			#expand("autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.log" , chrm=config["chromosome"]),
			expand("autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.evec", chrm=config["chromosome"]),
			expand("autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.eval", chrm=config["chromosome"]),
			expand("autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_Fix.evec", chrm=config["chromosome"]),
			expand("autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_Fix2.evec", chrm=config["chromosome"]),
			expand("autosomes/per_chr_results/chr{chrm}_inferred_pop_plot.pdf", chrm=config["chromosome"]),
			expand("autosomes/per_chr_results/chr{chrm}_inferred_pop_report.txt", chrm=config["chromosome"])


	# Keep only SNPs for both data sets by chromosome
	rule snps_ref_panel:
		input:
			ref = config["ref_path"],
			vcf_by_chr = (config["vcf_ref_panel_path"] + config["vcf_ref_panel_prefix"] + "{chrm}" + config["vcf_ref_panel_suffix"])
		output:
			"autosomes/ref_set/chr{chrm}_reference_panel_set_SNPs.recode.vcf"
		shell:
			"gatk -T SelectVariants -R {input.ref} -V {input.vcf_by_chr} -selectType SNP -o {output}"

	rule snps_unkn_set:
		input:
			vcf_by_chr = (config["vcf_unknown_set_path"] + config["vcf_unknown_set_prefix"] + "{chrm}" + config["vcf_unknown_set_suffix"])
		output:
			"autosomes/unk_set/chr{chrm}_unkown_set_SNPs.recode.vcf"
		shell:
			"vcftools --vcf {input.vcf_by_chr} --remove-indels --recode --recode-INFO-all -c > {output}"

	'''rule snps_unkn_set:
		input:
			ref = config["ref_path"],
			vcf_by_chr = (config["vcf_unknown_set_path"] + config["vcf_unknown_set_prefix"] + "{chrm}" + config["vcf_unknown_set_suffix"])
		output:
			"autosomes/unk_set/chr{chrm}_unkown_set_SNPs.recode.vcf"
		shell:
			"gatk -T SelectVariants -R {input.ref} -V {input.vcf_by_chr} -selectType SNP -o {output}"'''

	# Merge SNP data from both sets
	rule merge_snps:
		input:
			ref = config["ref_path"],
			vcf_unk_set = "autosomes/unk_set/chr{chrm}_unkown_set_SNPs.recode.vcf",
			vcf_ref_set = "autosomes/ref_set/chr{chrm}_reference_panel_set_SNPs.recode.vcf"
		output:
			"autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge.vcf"
		shell:
			"gatk -T CombineVariants -R {input.ref} --variant {input.vcf_ref_set} --variant {input.vcf_unk_set} -o {output} -genotypeMergeOptions UNIQUIFY"

	# Remove missing data
	rule rm_miss_data:
		input:
			"autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge.vcf"
		output:
			"autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing.recode.vcf"
		params:
			max_miss = config["genotype_call_rate_threshold"]
		shell:
			"vcftools --vcf {input} --max-missing {params.max_miss} --recode -c > {output}"


	# Convert to Plink
	rule conv_plink:
		input:
			"autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing.recode.vcf"
		output:
			merged_nomiss_plink_map = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.map",
			merged_nomiss_plink_ped = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.ped"
		params:
			chr_num = "{chrm}"
		shell:
			"vcftools --vcf {input} --plink --out autosomes/merge/chr{params.chr_num}_reference_panel_unknown_set_SNPs_merge_no_missing_plink"

	# LD Prune
	rule ld_prune:
		input:
	    	 plink_map_file = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.map",
	    	 plink_ped_file = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.ped"
		output:
	    	 in_prune = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.prune.in",
	    	 out_prune = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.prune.out"
		params:
			 chr_num = "{chrm}"
		shell:
			"plink --file autosomes/merge/chr{params.chr_num}_reference_panel_unknown_set_SNPs_merge_no_missing_plink --indep-pairwise 50 10 0.1 --out autosomes/merge/chr{params.chr_num}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune"

	rule rm_ld:
		input:
	    	 plink_map_file = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.map",
	    	 plink_ped_file = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink.ped",
			 out_prune = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.prune.out"
		output:
	    	 map_prune = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.map",
	    	 ped_prune = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.ped"
		params:
			chr_num = "{chrm}"
		shell:
			"plink --file autosomes/merge/chr{params.chr_num}_reference_panel_unknown_set_SNPs_merge_no_missing_plink --exclude {input.out_prune} --recode --out autosomes/merge/chr{params.chr_num}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune"

	# Next rules for all chromosomes here
	# Edit the 6th column of the ped files
	rule edit_ped_file:
		input:
			map_prune = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.map",
			ped_prune = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.ped"
		output:
			ped_fix = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_editColumn6.ped"
		shell:
			"""
			awk '{{$6 = "1"; print}}' {input.ped_prune} > {output.ped_fix}
			"""
	# Make par file
	rule make_par_file:
		input:
			map_file = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.map",
			ped_fix_file = "autosomes/merge/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_editColumn6.ped"
		output:
			"autosomes/pca/par/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_par_PCA.par"
		params:
			chr_num = "{chrm}"
		shell:
			"python make_par.py --map {input.map_file} --ped {input.ped_fix_file} --ev autosomes/pca/out/chr{params.chr_num}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune --par autosomes/pca/par/chr{params.chr_num}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_par"

	# Run smartpca
	rule run_pca:
		input:
			par = "autosomes/pca/par/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_par_PCA.par"
		output:
			evec = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.evec",
			eval_file = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.eval",
			#log_file = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.log"
		shell:"""
		smartpca -p {input.par}
		"""


	# Edit the evec files
	rule edit_evec_1:
		input:
			evec_file = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.evec"
		output:
			evec_fix = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_Fix.evec"
		params:
			chr_num = "{chrm}"
		shell: """awk '{{if($1 == "\\t" ) {{print $2,"\\t",$3,"\\t",$4,"\\t",$5,"\\t",$6,"\\t",$7,"\\t",$8,"\\t",$9,"\\t",$10,"\\t",$11,"\\t",$12,"\\t",$13,"\\t"}} else {{print $1,"\\t",$2,"\\t",$3,"\\t",$4,"\\t",$5,"\\t",$6,"\\t",$7,"\\t",$8,"\\t",$9,"\\t",$10,"\\t",$11,"\\t",$12,"\\t"}}}}' {input.evec_file} > {output.evec_fix}"""

	
	rule edit_evec_2:
		input:
			evec_fix_1 = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_Fix.evec"
		output:
			evec_fix_2 = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_Fix2.evec"
		shell: """awk '{{gsub(/\.variant2/,""); gsub(/\.variant/,""); print}}' {input.evec_fix_1} > {output.evec_fix_2}"""

	# Plot results and get inferred population report
	rule results:
		input:
			evec = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune_Fix2.evec",
			eval_file = "autosomes/pca/out/chr{chrm}_reference_panel_unknown_set_SNPs_merge_no_missing_plink_LDprune.eval",
			ref_panel = config["ref_panel_pop_info_path"],
			unk = config["unkn_panel_pop_info_path"]
		output:
			plot = "autosomes/per_chr_results/chr{chrm}_inferred_pop_plot.pdf",
			report = "autosomes/per_chr_results/chr{chrm}_inferred_pop_report.txt"
		params:
			chr_num = "{chrm}"
		shell:"""Rscript pca_inferred_ancestry_report.R {input.evec} {input.eval_file} {input.ref_panel} {input.unk} autosomes/per_chr_results/chr{params.chr_num}_inferred_pop_plot autosomes/per_chr_results/chr{params.chr_num}_inferred_pop_report"""


# PART 2: X CHROMOSOME
else:
	rule all:
		input:
			"chrX/ref_set/chrX_reference_panel_set_SNPs.recode.vcf",
			"chrX/unk_set/chrX_unknown_panel_set_SNPs.recode.vcf",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge.vcf",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR.vcf",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing.vcf",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink.map",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink.ped",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.map",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.ped",
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_editColumn6.ped",
			"chrX/pca/par/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_par_PCA.par",
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.evec",
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.eval",
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_Fix.evec",
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_Fix2.evec",
			"chrX/chrX_inferred_pop_plot.pdf",
			"chrX/chrX_inferred_pop_report.txt"

	# Keep the SNPS for each data set and biological sex
	rule keep_SNPS_chrX:
		input:
			ref = config["ref_path"],
			ref_panel = (config["vcf_ref_panel_path_X"] + config["vcf_ref_panel_file"]),
			unk_panel = (config["vcf_unknown_set_path_X"] + config["vcf_unknown_set_file"])
		output:
			ref_panel_SNPs = "chrX/ref_set/chrX_reference_panel_set_SNPs.recode.vcf",
			unk_panel_SNPs = "chrX/unk_set/chrX_unknown_panel_set_SNPs.recode.vcf"
		shell: """
			gatk -T SelectVariants -R {input.ref} -V {input.ref_panel} -selectType SNP -o {output.ref_panel_SNPs};
			vcftools --vcf {input.unk_panel} --remove-indels --recode --recode-INFO-all -c > {output.unk_panel_SNPs}
		"""

	# Merge the files
	rule merge_chrX_files:
		input:
			ref = config["ref_path"],
			ref_panel = "chrX/ref_set/chrX_reference_panel_set_SNPs.recode.vcf",
			unk_panel = "chrX/unk_set/chrX_unknown_panel_set_SNPs.recode.vcf"
		output:
			merge_ref_unk = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge.vcf"
		shell:
			"gatk -T CombineVariants -R {input.ref} --variant {input.ref_panel} --variant {input.unk_panel} -o {output.merge_ref_unk} -genotypeMergeOptions UNIQUIFY"

	# Remove the PARS and X Transposed regions
	rule remove_PARS_XTR:
		input:
			merge_ref_unk = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge.vcf",
			coordinates = config["X_chr_coordinates"]
		output:
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR.vcf"
		shell:
			"bedtools subtract -header -a {input.merge_ref_unk} -b {input.coordinates} > {output}"

	# Remove missing data
	rule remove_missing_data_X:
		input:
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR.vcf"
		output:
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing.vcf"
		params:
			max_miss = config["genotype_call_rate_threshold"]
		shell:
			"vcftools --vcf {input} --max-missing {params.max_miss} --recode -c > {output}"

	# Convert to Plink
	rule convert_plink_X:
		input:
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing.vcf"
		output:
			ref_unk_plink_map = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink.map",
			ref_unk_plink_ped = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink.ped"
		shell:
			"vcftools --vcf {input} --plink --out chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink"

	# LD Prune
	rule ld_prune_x:
		input:
			ref_unk_plink_map = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink.map",
			ref_unk_plink_ped = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink.ped"
		output:
			ref_unk_map_prune = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.map",
			ref_unk_ped_prune = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.ped"
		shell:
			"""
			plink --file chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink --ld-xchr 1 --recode --out chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune
			"""

	# Edit the 6th column of the .ped file
	rule edit_ped_file:
		input:
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.ped"
		output:
			"chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_editColumn6.ped"
		shell: """
			awk '{{$6 = "1"; print}}' {input} > {output}
		"""

	# Make the par file
	rule make_par_file:
		input:
			ref_unk_map = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.map",
			ref_unk_ped_edit = "chrX/merge/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_editColumn6.ped"
		output:
			"chrX/pca/par/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_par_PCA.par"
		shell:
			"python make_par.py --map {input.ref_unk_map} --ped {input.ref_unk_ped_edit} --ev chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune --par chrX/pca/par/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_par"

	# run PCA
	rule run_pca:
		input:
			par = "chrX/pca/par/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_par_PCA.par"
		output:
			evec = "chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.evec",
			eval_file = "chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.eval",
		shell:"""
		smartpca -p {input.par}
		"""
	
	# Edit the evec files
	rule edit_evec_1:
		input:
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.evec"
		output:
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_Fix.evec"
		shell:"""awk '{{if($1 == "\\t" ) {{print $2,"\\t",$3,"\\t",$4,"\\t",$5,"\\t",$6,"\\t",$7,"\\t",$8,"\\t",$9,"\\t",$10,"\\t",$11,"\\t",$12,"\\t",$13,"\\t"}} else {{print $1,"\\t",$2,"\\t",$3,"\\t",$4,"\\t",$5,"\\t",$6,"\\t",$7,"\\t",$8,"\\t",$9,"\\t",$10,"\\t",$11,"\\t",$12,"\\t"}}}}' {input} > {output}"""

	rule edit_evec_2:
		input:
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_Fix.evec"
		output:
			"chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_Fix2.evec"
		shell:"""awk '{{gsub(/\.variant2/,""); gsub(/\.variant/,""); print}}' {input} > {output}"""

	# Plot results and get inferred population report
	rule results:
		input:
			evec = "chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune_Fix2.evec",
			eval_file = "chrX/pca/out/chrX_reference_panel_unknown_set_SNPs_merge_noPARS_noXTR_noMissing_plink_LDprune.eval",
			ref_panel = config["ref_panel_pop_info_path"],
			unk = config["unkn_panel_pop_info_path"]
		output:
			plot = "chrX/chrX_inferred_pop_plot.pdf",
			report = "chrX/chrX_inferred_pop_report.txt"
		shell:"""Rscript pca_inferred_ancestry_report.R {input.evec} {input.eval_file} {input.ref_panel} {input.unk} chrX/chrX_inferred_pop_plot chrX/chrX_inferred_pop_report"""	
