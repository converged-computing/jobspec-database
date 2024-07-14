## Sylvain SCHMITT
## 28/04/2021

configfile: "config/config.dag.yml"
libraries, = glob_wildcards(config["libdir"] + "/{library}_mutated_R1.fastq")

rule all:
    input:
        expand("results/mutations/{library}_{caller}.vcf", library=libraries, caller=config["callers"]), # mutations
        expand("results/stats/{library}_{caller}.tsv", library=libraries, caller=config["callers"]) # stats
        # expand("results/{library}/multiqc_report.html", library=libraries)

# Rules #

## Reference ##
include: "rules/bwa_index.smk"
include: "rules/samtools_faidx.smk"
include: "rules/gatk_dict.smk"
include: "rules/gatk_idx.smk"

## Reads ##
include: "rules/fastqc.smk"
include: "rules/trimmomatic.smk"

## Alignments ##
include: "rules/bwa_mem.smk"
include: "rules/samtools_sort.smk"
include: "rules/samtools_index.smk"
include: "rules/gatk_markduplicates.smk"
include: "rules/samtools_index_md.smk"
include: "rules/samtools_mpileup.smk"
include: "rules/samtools_stats.smk"
include: "rules/qualimap.smk"

## Detection ##

include: "rules/gatk_mutect2.smk"
include: "rules/freebayes.smk"
include: "rules/gatk_haplotypecaller.smk"
include: "rules/gatk_genotypegvcfs.smk"
include: "rules/strelka2.smk"
include: "rules/varscan.smk"
include: "rules/varscan2vcf.smk"
include: "rules/somaticsniper.smk"
include: "rules/muse.smk"

## Mutations ##
include: "rules/bedtools_substract.smk"
include: "rules/cp_vcfs.smk"

## QC ##
include: "rules/multiqc.smk"
include: "rules/evaluate_call.smk"
