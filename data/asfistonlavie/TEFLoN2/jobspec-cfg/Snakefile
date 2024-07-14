#import dependences python
import subprocess
import os
import sys

#function
def get_mem_mb(wildcards, attempt):
	return config["PARAMS"]["GENERAL"]["MEMORY"] +  (config["PARAMS"]["GENERAL"]["MEMORY_SUPP"] * (attempt - 1))



# check if varaible is None or empty
def check_value (var):
	if var is None :
		return False
	elif type(var) is int:
		return True
	elif type(var) is str :
		if len(var.strip()) == 0 :
			return False
		else :
			return True


if (config["PARAMS"]["FILTERFASTQ"]["USE"] == 1)  :
	def reads1_fastp (wcs):
		name = wcs.reads
		wd = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/reads1/"
		for file_ext in [".fastq", ".fq"] :
			for file_r1 in ["_1","_r1","_R1",".1",".r1",".R1"] :
				file = wd + name + file_r1 +file_ext
				if ( os.path.exists(file)  or os.path.exists(file + ".gz")) :
					read1 = wd + name + file_r1 + file_ext + ".gz"
					return read1
		read1 = wd + name + "_1.fastq.gz"
		return read1

	def reads2_fastp (wcs):
		name = wcs.reads
		wd = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/reads2/"
		for file_ext in [".fastq", ".fq"] :
			for file_r2 in ["_2","_r2","_R2",".2",".r2",".R2"] :
				file = wd + name + file_r2 +file_ext
				if ( os.path.exists(file)  or os.path.exists(file + ".gz")) :
					read2 = wd + name + file_r2 + file_ext + ".gz"
					return read2
		read2 = wd + name + "_2.fastq.gz"
		return read2


	def reads1 (wcs):
		name = wcs.reads
		wd = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/reads1/"
		read1 = wd + name + "_1.filter.fastq.gz"
		return read1

	def reads2 (wcs):
		name = wcs.reads
		wd = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/reads2/"
		read2 = wd + name + "_2.filter.fastq.gz"
		return read2

else :
	def reads1 (wcs):
		name = wcs.reads
		wd = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/reads1/"
		for file_ext in [".fastq", ".fq"] :
			for file_r1 in ["_1","_r1","_R1",".1",".r1",".R1"] :
				file = wd + name + file_r1 +file_ext
				if ( os.path.exists(file)  or os.path.exists(file + ".gz")) :
					read1 = wd + name + file_r1 + file_ext + ".gz"
					return read1
		read1 = wd + name + "_1.fastq.gz"
		return read1

	def reads2 (wcs):
		name = wcs.reads
		wd = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/reads2/"
		for file_ext in [".fastq", ".fq"] :
			for file_r2 in ["_2","_r2","_R2",".2",".r2",".R2"] :
				file = wd + name + file_r2 +file_ext
				if ( os.path.exists(file)  or os.path.exists(file + ".gz")) :
					read2 = wd + name + file_r2 + file_ext + ".gz"
					return read2
		read2 = wd + name + "_2.fastq.gz"
		return read2



def reads_compress (wcs):
	name = wcs.reads
	wd = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/"
	for folder in ["reads1/", "reads2/"] :
		if os.path.exists(wd + folder + name) :
			read = wd + folder + name
			return read


def samples_list() :
	base = config["DATA_INPUT"]["WORKING_DIRECTORY"] + "/samples/"
	folders_reads = ["reads1/","reads2/"]
	folders_for_formating = ["bam/","reads/"] 
	wd = []
	samples = {"id":[]}
	for i in folders_reads :
		wd = base + i
		list_samples = os.listdir(os.path.abspath(wd))
		for element in list_samples:
			if ".fastq" in element or ".fq" in element :
				if ".gz" in element:
					id = element.rsplit(".",2)[0]
				else:
					id = element.rsplit(".",1)[0]
				ext_underscore = id.rsplit("_",1)
				ext_point = id.rsplit(".",1)
				if (len(ext_underscore) == 2 and (ext_underscore[1] == "1" or ext_underscore[1] == "2" or ext_underscore[1] == "r1" or ext_underscore[1] == "r2" or ext_underscore[1] == "R1" or ext_underscore[1] == "R2")) :
					samples["id"].append(ext_underscore[0])
				if (len(ext_point) == 2 and (ext_point[1] == "1" or ext_point[1] == "2" or ext_point[1] == "r1" or ext_point[1] == "r2" or ext_underscore[1] == "R1" or ext_underscore[1] == "R2")) :
					samples["id"].append(ext_point[0])

	for i in folders_for_formating:
		wd = base + i
		list_samples = os.listdir(os.path.abspath(wd))
		for element in list_samples:
			if ".fastq" in element or ".fq" in element or ".bam" in element:
				if ".gz" in element:
					id = element.rsplit(".",2)[0]
				else:
					id = element.rsplit(".",1)[0]
				samples["id"].append(id)

	return samples


def change_wildcard_for_input(input_files,samples_all,str_replace) :
	input_files_all = []
	for element in input_files:
		for sample in samples_all:
			tmp = element.replace(str_replace,sample)
			input_files_all.append(tmp)
	return input_files_all


if config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"].strip() == "" :
	config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"] = "data_output_"
else :
	config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"] = config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+"/data_output_"

localrules:
	all

#rule all defines the output of the pipeline
##global variable

wd = config["DATA_INPUT"]["WORKING_DIRECTORY"]
dict_samples = samples_list()
samples_all = set(dict_samples["id"])

include: "modules/formatting.smk"
include: "modules/bamtofastq.smk"

#Choice of the script to call according to the input data file
use_custom = 0 # 0: use annotation, 1:use custom
if check_value(config["DATA_INPUT"]["GENOME"]):
	if check_value(config["DATA_INPUT"]["ANNOTATION"]):
		include: "modules/teflon_prep_annotation.smk"
	else :
		use_custom = 1
		include: "modules/teflon_prep_custom.smk"
else : 
	sys.exit("Invalid inputs")

if (config["PARAMS"]["FILTERFASTQ"]["USE"] == 1)  :
	include: "modules/filterfastq.smk"


#Calling snakemake modules
include: "modules/mapping.smk"
include: "modules/teflon_discover.smk"
include: "modules/subsample.smk"
include: "modules/teflon_collapse.smk"
include: "modules/preliminaryResults.smk"
include: "modules/teflon_count.smk"
include: "modules/teflon_genotype.smk"

input_files = []

if (use_custom == 0 and config["PARAMS"]["PREP_ANNOTATION"]["STOP"] == 1) :
			input_files = [
				config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/0-reference/"+config["PARAMS"]["GENERAL"]["PREFIX"]+".prep_MP/"+config["PARAMS"]["GENERAL"]["PREFIX"]+".mappingRef.fa.bwt"
				]

elif (use_custom == 1 and config["PARAMS"]["PREP_CUSTOM"]["STOP"] == 1) :
		input_files = [
			config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/0-reference/"+config["PARAMS"]["GENERAL"]["PREFIX"]+".prep_MP/"+config["PARAMS"]["GENERAL"]["PREFIX"]+".mappingRef.fa.bwt"
			]

elif (config["PARAMS"]["DISCOVER"]["STOP"] == 1) :
	input_files_tmp = [
		config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/1-mapping/{samples_all}.sorted.cov.txt",
		config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/1-mapping/{samples_all}.sorted.stats.txt",
		config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/3-countPos/{samples_all}.all_positions_sorted.txt",
		config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/3-countPos/{samples_all}.all_positions.txt"
		]
	input_files = change_wildcard_for_input(input_files_tmp,samples_all,"{samples_all}")

elif (config["PARAMS"]["COLLAPSE"]["STOP"] == 1) :	
	input_files = [config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/2-preliminaryResults/TE_catalog.summary"]

elif (config["PARAMS"]["COUNT"]["STOP"] == 1) :
	input_files_tmp = [config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/3-countPos/{samples_all}.counts.txt"]
	input_files = change_wildcard_for_input(input_files_tmp,samples_all,"{samples_all}")

elif check_value(config["PARAMS"]["GENOTYPE"]["POPULATION"]["FILE"]) :
	popFILE = config["PARAMS"]["GENOTYPE"]["POPULATION"]["FILE"]
	group = []
	with open(popFILE,"r") as fIN:
		for line in fIN:
			if line.endswith("\n") :
				fields = line[:-1].split("\t")
			else : 
				fields = line.split("\t")
			group.append(fields[1])
	group = set(group)
	include: "modules/teflon_genotype_pop.smk"
	input_files = [config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/4-genotypes/populations/all_frequency.population.genotypes.txt",config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/4-genotypes/populations/all_frequency.population.genotypes2.txt"]

else :
	input_files = [config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/4-genotypes/samples/"+"all_samples.genotypes.txt",config["PARAMS"]["GENERAL"]["WORKING_DIRECTORY"]+config["PARAMS"]["GENERAL"]["PREFIX"]+"/4-genotypes/samples/"+"all_samples.genotypes2.txt"]
	
rule all:
	input:
		input_files