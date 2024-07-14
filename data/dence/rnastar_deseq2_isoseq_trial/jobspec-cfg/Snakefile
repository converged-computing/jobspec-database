##### load rules #####

include: "rules/common.smk"
include: "rules/trim.smk"
include: "rules/align.smk"
include: "rules/diffexp.smk"
include: "rules/samtools_index.smk"
include: "rules/aligned_report.smk"
#include: "rules/qc.smk"

import pandas as pd
from snakemake.utils import validate, min_version
##### set minimum snakemake version #####
min_version("5.1.2")

##### target rules #####

def all_input(wildcards):
	wanted_input = []

	wanted_input.extend(
		expand(["results/star/{sample}-{unit}.Aligned.sortedByCoord.out.bam"],
			sample=units["sample"],unit=units["unit"]))
	#wanted_input.extend(
	#		expand(["results/diffexp/{contrast}.diffexp.tsv",
	#			"results/diffexp/{contrast}.ma-plot.svg"],
	#			contrast=config["diffexp"]["contrasts"]))
	#wanted_input.extend(["results/pca.svg"])
	wanted_input.extend(["results/reports/star_percent_aligned_report.txt"])
	wanted_input.extend(["results/reports/spike_in_aligned_report.txt"])
	wanted_input.extend(["results/reports/spike_in_coefficient_of_deviation.txt"])
	#wanted_input.extend(["results/pca.svg","qc/multiqc_report.html"])
	return wanted_input

rule all:
	input: all_input

##### setup singularity #####

# this container defines the underlying OS for each job when using the workflow
# with --use-conda --use-singularity
singularity: "docker://continuumio/miniconda3"

##### setup report #####
report: "report/workflow.rst"
