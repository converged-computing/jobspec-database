# Snakemake pipeline for the analysis of ChIP_seq PE and SE


################## Import libraries ##################

import pandas as pd
import os
import sys
from subprocess import call
import itertools
from snakemake.utils import R


################## Configuration file ##################

configfile: "config.yaml"
singularity:'docker://biocontainers/bowtie2:v2.4.1_cv1'

WORKING_DIR =   config["working_dir"]
RESULT_DIR  =   config["result_dir"]
annotation  =   config["annotation"]

################## Configuration file ##################

# read the tab separated table containing columns: sample, fq1, fq2 and condition
units = pd.read_table(config["units"], dtype=str).set_index(["sample"], drop=False)
SAMPLES = units.index.get_level_values('sample').unique().tolist()
samples = pd.read_csv(config["units"], dtype=str,index_col=0,sep="\t")
samplefile = config["units"]

################## Helper functions ##################

def sample_is_single_end(sample):
    """This function detect missing value in the column 2 of the units.tsv"""
    if "fq2" not in samples.columns:
        return True
    else:
        return pd.isnull(samples.loc[(sample), "fq2"])

def get_fastq(wildcards):
    """This function checks if the sample has paired end or single end reads and returns 1 or 2 names of the fastq files"""
    if sample_is_single_end(wildcards.sample):
        return samples.loc[(wildcards.sample), ["fq1"]].dropna()
    else:
        return samples.loc[(wildcards.sample), ["fq1", "fq2"]].dropna()

def get_trim_names(wildcards):
    """
    This function:
      1. Checks if the sample is paired end or single end
      2. Returns the correct input and output trimmed file names. 
    """
    if sample_is_single_end(wildcards.sample):
        inFile = samples.loc[(wildcards.sample), ["fq1"]].dropna()
        return "--in1 " + inFile[0] + " --out1 " + WORKING_DIR + "trimmed/" + wildcards.sample + "_R1_trimmed.fq.gz" 
    else:
        inFile = samples.loc[(wildcards.sample), ["fq1", "fq2"]].dropna()
        return "--in1 " + inFile[0] + " --in2 " + inFile[1] + " --out1 " + WORKING_DIR + "trimmed/" + wildcards.sample + "_R1_trimmed.fq.gz --out2 "  + WORKING_DIR + "trimmed/" + wildcards.sample + "_R2_trimmed.fq.gz"

def get_star_names(wildcards):
    """
    This function:
      1. Checks if the sample is paired end or single end.
      2. Returns the correct input file names for STAR mapping step.
    """
    if sample_is_single_end(wildcards.sample):
        return WORKING_DIR + "trimmed/" + wildcards.sample + "_R1_trimmed.fq.gz"     
    else:
        return WORKING_DIR + "trimmed/" + wildcards.sample + "_R1_trimmed.fq.gz " + WORKING_DIR + "trimmed/" + wildcards.sample + "_R2_trimmed.fq.gz"


def get_samples_per_treatment(input_df="units.tsv",colsamples="sample",coltreatment="condition",treatment="control"):
    """This function returns a list of samples that correspond to the same experimental condition"""
    df = pd.read_table(input_df)
    df = df.loc[df[coltreatment] == treatment]
    filtered_samples = df[colsamples].tolist()
    return filtered_samples

CASES = get_samples_per_treatment(treatment="treatment")
CONTROLS = get_samples_per_treatment(treatment="control")
################## Wilcards constrains  ##################

wildcard_constraints:
    sample = "[A-Za-z0-9]+"

wildcard_constraints:
    unit = "[A-Za-z0-9]+"

################## DESIRED OUTPUT ##################
##############
# Desired output
##############
FASTP               =     expand(WORKING_DIR + "trimmed/" + "{sample}_{read}_trimmed.fq.gz", sample=SAMPLES, read={"R1", "R2"})
BOWTIE2             =     expand(WORKING_DIR + "mapped/{sample}.bam", sample= SAMPLES)
FASTQC              =     expand(RESULT_DIR + "fastqc/{sample}.fastqc.html", sample=SAMPLES)
BIGWIG              =     expand(RESULT_DIR + "bigwig/{sample}_rpkm.bw", sample=SAMPLES)
MACS2               =     expand(RESULT_DIR + "macs2/{treatment}_vs_{control}_peaks.narrowPeak", treatment = CASES, control = CONTROLS)  

###############
# Final output
################
rule all:
    input:
        FASTP,
        BOWTIE2,
        BIGWIG,
        MACS2
        
    message: "ChIP-seq SE pipeline succesfully run."		#finger crossed to see this message!
    singularity:'docker://ewels/multiqc:v1.13'
    shell:"""
    cp units.tsv results/ 
    cp config.yaml results/
    multiqc *  --outdir results/
    #rm -rf {WORKING_DIR}
    """

###############
# Rules
###############

include : "rules/external_data.smk"
include : 'rules/pre_processing.smk'
include : "rules/peak_calling.smk"
include : "rules/deeptools_post_processing.smk"

