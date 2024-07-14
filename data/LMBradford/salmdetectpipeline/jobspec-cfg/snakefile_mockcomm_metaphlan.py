#NOTES##############################################################################################
#No conda on Compute Canada
##Software loaded as modules in SLURM file instead

# Submit as one big job.
# Metaphlan4 database is in the scratch directory


#CONFIG FILE ###########################################################################################################################################################

#Define the config yaml file in the command line

#VARIABLE SETUP ########################################################################################################################################################

#Import packages for file name management
import glob
#These imports let me use wildcards to take in file names and maintain the name info throughout the workflow

import os

#Detect wildcards from input files names
#
filename = config['pathtoinput'] + "{name}_R1.fq.gz"
NAME, = glob_wildcards(filename)
print(NAME)


#CATCH-ALL RULE ###################################################################################################################################################

rule all:
  #List all the final output files expected here.
	input:
		config['outputdir'] + 'metaphlan4_profiles.txt',
		expand(config['outputdir'] + '{name}_profiled_metagenome.txt', name=NAME),
		expand(config['outputdir'] + 'Salmhits/{name}_salmhits.txt', name=NAME),   
		

#COMBINE FWD, REV, UNPAIRED READS #############################################################

rule combine:
	input:
		fwd = "Trim/{name}_R1.P.fq.gz",
		rev = "Trim/{name}_R2.P.fq.gz",
		unpair = "Trim/{name}_unpaired.fq.gz"
	output: temp(config['outputdir'] + '{name}_combo.fastq.gz')
	shell: 'cat {input.fwd} {input.rev} {input.unpair} > {output}'

#RUN METAPHLAN4 ###########################################################################

rule metaphlan:
	input:
		reads = rules.combine.output
	output: 
		profile = config['outputdir'] + '{name}_profiled_metagenome.txt',
		bt2 = config['outputdir'] + '{name}.bowtie2.bz2'
	params:
		index = config['index'],
		db = config['dbloc'],
	threads: 4 
	shell: 'metaphlan {input} --input_type fastq \
	-o {output.profile} \
	--bowtie2out {output.bt2} \
	--index {params.index} \
	--bowtie2db {params.db} \
  --nproc {threads}   \
  --unclassified_estimation \
  -t rel_ab_w_read_stats'

#COMBINE PROFILES ##########################################################################

rule mergeprofiles:
	input: expand(config['outputdir'] + '{name}_profiled_metagenome.txt', name=NAME)
	output: config['outputdir'] + 'metaphlan4_profiles.txt'
	shell:
		'merge_metaphlan_tables.py {input} > {output}'

#Extract Salmonella hits ###################################################################
rule extsalmhits:
	input: config['outputdir'] + '{name}.bowtie2.bz2'
	output: config['outputdir'] + 'Salmhits/{name}_salmhits.txt'
	params:
		salmmarkers = config['salmmarkerlist']
	shell:
		'bzcat {input} | grep -F -f {params.salmmarkers} > {output}'
